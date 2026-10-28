"""VimBlock: one bidirectional Vision Mamba block (the whole of Algorithm 1).

What this file does:
    Takes a sequence of 65 tokens, normalises it, widens it, runs a selective
    state space model over it TWICE (once front-to-back, once back-to-front,
    each with its own weights), gates and adds the two results, projects back
    down, and adds the result onto the input (a residual connection).

Vim paper correspondence (arXiv 2401.09417, Section 3.3 and Algorithm 1, page 6):
    Algorithm 1 in plain English, line by line, with where each line lives.
    Input: token sequence T_{l-1} of shape (B, M, D) = (B, 65, 192).
    Output: token sequence T_l, same shape.

     1  (comment) normalise the input sequence
     2  T' = Norm(T_{l-1})                     -> VimBlock.forward, self.norm
     3  x = Linear^x(T')      (B, M, E)        -> VimBlock.forward, first half of
     4  z = Linear^z(T')      (B, M, E)           self.in_proj (split explicitly)
     5  (comment) process with different direction
     6  for o in {forward, backward}:          -> VimBlock.forward, the `for` loop,
                                                   using self.forward_ssm and
                                                   self.backward_ssm
     7      x'_o = SiLU(Conv1d_o(x))           -> DirectionalSSM.forward
     8      B_o = Linear^B_o(x'_o)  (B, M, N)  -> DirectionalSSM.forward, proj_B
     9      C_o = Linear^C_o(x'_o)  (B, M, N)  -> DirectionalSSM.forward, proj_C
    10      (comment) softplus ensures positive Delta_o
    11      Delta_o = log(1 + exp(Linear^Delta_o(x'_o) + Parameter^Delta_o))
                                                -> proj_delta + delta_bias, softplus
    12      (comment) shape of Parameter^A_o is (E, N)
                                                -> A = -exp(A_log)
    13      A_bar_o = Delta_o (x) Parameter^A_o   -> vim/selective_scan.py
    14      B_bar_o = Delta_o (x) B_o             -> vim/selective_scan.py
   15-22    initialise h and y, run the SSM recurrence over i = 0 .. M-1
                                                -> vim/selective_scan.py
    23  end for
    24  (comment) get gated y_o
    25  y'_forward  = y_forward  (.) SiLU(z)   -> VimBlock.forward, after the loop
    26  y'_backward = y_backward (.) SiLU(z)   -> VimBlock.forward, after the loop
    27  (comment) residual connection
    28  T_l = Linear^T(y'_forward + y'_backward) + T_{l-1}
                                                -> self.out_proj, then `+ residual`

    For the backward direction, "Conv1d_o" and the scan run on the REVERSED
    sequence. We implement that by flipping the input, running the very same
    code, and flipping the output back.

Shapes (defaults): (B, 65, 192) -> (B, 65, 192), with E = 384 and N = 16 inside.

Deliberate differences from the paper / the official code:
    1. Delta's projection is a full Linear E -> E (384 x 384 weights), as PRINTED
       in Algorithm 1 line 11. The official Mamba/Vim code instead uses a
       low-rank factorisation (E -> 12 -> E) to save parameters. We follow the
       printed algorithm; vim/vision_mamba.py discusses the parameter cost.
    2. Norm is nn.LayerNorm. The paper just says "Norm"; the official Vim-Ti
       code uses RMSNorm, a slightly simpler variant (no mean subtraction).
    3. The official Mamba code adds a learned skip term D * x to the scan output
       and the official Vim code divides the sum of the two directions by 2.
       Neither appears in the printed Algorithm 1, so neither is here.
    4. The selective scan is a slow Python loop instead of a fused CUDA kernel
       (paper Section 3.5, out of scope; see vim/selective_scan.py).

Observation worth knowing (measured in tests/test_vim_block.py):
    With the small initial step sizes (Delta in 0.001..0.1) and no D * x skip
    term, a freshly initialised block adds very little to its input (the
    added part has a standard deviation around 0.0004 for an input of 1). So a
    new 12-block stack starts out close to the identity function. That is
    harmless, and even helpful: the deep stack is perfectly stable at the
    start, and AdamW sizes each parameter's update by the gradient's RELATIVE
    size, so small gradients still produce useful steps and training grows
    the branches. (The official code's D * x term would make the branch
    output larger from step one.)

Why bidirectional at all (paper Section 3.3, Table 4):
    A single forward scan carries information only from left to right. Token
    32 (the class token) would never hear from tokens 33..64, the second half
    of the image! For a 1D text sequence that direction is natural, but an
    image has no "start" and "end". So Vim runs a second, backward scan, and
    every token, the class token included, gets information from both sides.
    Paper Table 4: without the backward pass ("None") 73.2% ImageNet top-1 and
    32.3 ADE20K mIoU; with the default bidirectional SSM + Conv1d 73.9% and
    35.9 mIoU.
"""

import math  # standard library maths (log for the Delta initialisation)

import torch  # main PyTorch library (tensors)
import torch.nn as nn  # neural network building blocks (Linear, Conv1d, LayerNorm)

# `torch.nn.functional` holds the same operations as `torch.nn`, but as plain
# FUNCTIONS with no stored weights (e.g. F.silu(x)). Layers WITH weights are
# objects from torch.nn (nn.Linear); weightless operations are calls from F.
import torch.nn.functional as F

# `.selective_scan` with a leading dot means "the selective_scan.py file in the
# SAME package (folder) as this file", i.e. vim/selective_scan.py.
from .selective_scan import selective_scan


class DirectionalSSM(nn.Module):
    """Algorithm 1 lines 7-22 for ONE direction: conv1d + SiLU, B, C, Delta, A, scan.

    VimBlock owns two of these, one per direction, so every weight here exists
    twice: once for the forward scan and once for the backward scan.

    Shapes (defaults): x (B, 65, 384) -> y (B, 65, 384)
    """

    def __init__(self, expand_dim=384, d_state=16, conv_kernel=4, dt_min=0.001,
                 dt_max=0.1):
        # Parent constructor first, so nn.Module can register our layers
        super().__init__()

        # Remember E for the shape checks in forward()
        self.expand_dim = expand_dim

        # -----------------------------------------------------------------------------
        # Algorithm 1, line 7: the depthwise causal Conv1d
        # -----------------------------------------------------------------------------
        # A 1D convolution slides a small window (kernel_size = 4 tokens) along the
        # sequence and, at each position, outputs a weighted sum of the tokens in
        # the window. This gives each token a little LOCAL context (its 3
        # neighbours) before the long-range note passing of the scan begins.
        #
        # groups=expand_dim makes it DEPTHWISE: the 384 channels are split into 384
        # groups of one, so channel e only ever looks at channel e of the
        # neighbouring tokens (4 weights + 1 bias per channel). A normal conv would
        # mix all 384 channels together, needing 384 * 384 * 4 weights. Mixing
        # channels is the job of the Linear layers; the conv only mixes positions.
        #
        # padding=conv_kernel-1 (= 3) adds 3 zeros at BOTH ends of the sequence. The
        # output is then 3 longer than the input, and forward() crops the extra 3
        # off the END. The result is CAUSAL: the output at token t is built from
        # tokens t-3, t-2, t-1 and t only, never from the future. That matches a
        # scan, which also only knows the past at step t.
        self.conv1d = nn.Conv1d(
            in_channels=expand_dim,
            out_channels=expand_dim,
            kernel_size=conv_kernel,
            groups=expand_dim,
            padding=conv_kernel - 1,
        )

        # -----------------------------------------------------------------------------
        # Algorithm 1, lines 8-9: B and C, computed from each token (the "selective"
        # part). Each token's 384 numbers are mapped to N = 16 numbers.
        # bias=False follows the official Mamba code (its projection has no bias).
        # -----------------------------------------------------------------------------
        # nn.Linear(in, out) is a learnable matrix multiply y = x W^T + b. Applied to
        # a (B, 65, 384) tensor it acts on the LAST dimension only, i.e. separately
        # for each of the B * 65 tokens: (B, 65, 384) -> (B, 65, 16).
        self.proj_B = nn.Linear(expand_dim, d_state, bias=False)
        self.proj_C = nn.Linear(expand_dim, d_state, bias=False)

        # -----------------------------------------------------------------------------
        # Algorithm 1, line 11: Delta = softplus(Linear^Delta(x') + Parameter^Delta)
        # -----------------------------------------------------------------------------
        # The printed algorithm projects E -> E directly (384 x 384 weights), and we
        # follow it. (The official Mamba code factorises this into E -> 12 -> E.)
        # The bias is a separate nn.Parameter, exactly as the algorithm writes it
        # ("Parameter^Delta"), so we can give it a special initialisation below.
        self.proj_delta = nn.Linear(expand_dim, expand_dim, bias=False)
        self.delta_bias = nn.Parameter(torch.zeros(expand_dim))

        # -----------------------------------------------------------------------------
        # Algorithm 1, line 12: Parameter^A of shape (E, N), stored as its log
        # -----------------------------------------------------------------------------
        # We never store A itself. We store A_log and use A = -exp(A_log). Since
        # exp(anything) > 0, A = -exp(A_log) < 0 for EVERY value the optimiser could
        # ever choose for A_log. A negative A keeps A_bar = exp(Delta * A) in (0, 1),
        # so the scan's memory always decays and never explodes (see
        # vim/selective_scan.py). If we stored A directly, one large gradient step
        # could push an entry above zero and make training blow up.
        self.A_log = nn.Parameter(torch.zeros(expand_dim, d_state))

        # Fill the parameters with their starting values
        self._init_weights(d_state, dt_min, dt_max)

    def _init_weights(self, d_state, dt_min, dt_max):
        # torch.no_grad(): we are setting starting values, not computing anything
        # we want gradients for.
        # PYTHON CONCEPT: a `with` block can also switch a mode on for the duration
        # of the block and switch it off again afterwards (here: gradient tracking).
        with torch.no_grad():
            # --- A: the "S4D-real" initialisation ---
            # A[e, n] = -(n + 1), i.e. each channel's 16 slots get decay rates
            # -1, -2, ..., -16. torch.arange(1, 17) = [1, 2, ..., 16], shape (16,).
            # Slot 1 leaks slowly (long memory), slot 16 leaks fast (short memory),
            # so from the very first step the model has a SPREAD of time scales to
            # choose from. The same 16 rates are used for all 384 channels.
            rates = torch.arange(1, d_state + 1, dtype=torch.float32)
            # (16,) -> (1, 16) via unsqueeze(0), then broadcast into A_log (384, 16).
            # copy_ writes in place into the existing Parameter (keeps it registered).
            self.A_log.copy_(torch.log(rates).unsqueeze(0).expand_as(self.A_log))

            # --- Delta bias: start each channel at a step size in [dt_min, dt_max] ---
            # Draw one target step size per channel, spread evenly on a LOG scale
            # between 0.001 and 0.1 (so 0.001-0.01 is as likely as 0.01-0.1).
            # This is the Mamba initialisation. Small step sizes mean A_bar starts
            # close to 1: at the start of training, the memory lasts a long time.
            log_min = math.log(dt_min)
            log_max = math.log(dt_max)
            # torch.rand gives uniform numbers in [0, 1); scale them into
            # [log_min, log_max), then exp() takes them back to [dt_min, dt_max).
            uniform = torch.rand(self.expand_dim)
            dt = torch.exp(uniform * (log_max - log_min) + log_min)
            # We want softplus(delta_bias) = dt when the projection outputs 0, so set
            # delta_bias to the INVERSE of softplus: if softplus(b) = log(1 + e^b)
            # = dt, then b = log(e^dt - 1). torch.expm1(dt) computes e^dt - 1
            # accurately even when dt is tiny (where e^dt is very close to 1).
            self.delta_bias.copy_(torch.log(torch.expm1(dt)))

        # The Linear and Conv1d layers keep PyTorch's default initialisation
        # (uniform in +-1/sqrt(fan_in)). For proj_delta that is +-1/sqrt(384) = 0.051,
        # which is the same scale rule the Mamba code uses for its Delta projection.

    def forward(self, x):
        # x: (B, 65, 384), the "x" half of in_proj, already flipped by the caller
        # for the backward direction.
        # Check that the channel count matches what we were built for
        assert x.shape[2] == self.expand_dim, (
            f"Expected {self.expand_dim} channels, got {x.shape[2]}"
        )
        # M, the sequence length, needed to crop the conv output back
        seq_len = x.shape[1]

        # ----- Algorithm 1, line 7: x' = SiLU(Conv1d(x)) -----

        # nn.Conv1d wants (batch, channels, length), but tokens are stored as
        # (batch, length, channels). Swap the last two axes:
        #   (B, 65, 384) -> (B, 384, 65)
        x_conv = x.transpose(1, 2)

        # Run the depthwise conv. With 3 zeros of padding at each end the output is
        # 65 + 2*3 - 4 + 1 = 68 long:  (B, 384, 65) -> (B, 384, 68)
        x_conv = self.conv1d(x_conv)

        # Keep the FIRST 65 outputs and drop the last 3. Output position t then
        # covers input tokens t-3 .. t (the padding zeros stand in for "before the
        # start"), so no token can peek at a later one: the conv is causal.
        #   (B, 384, 68) -> (B, 384, 65)
        x_conv = x_conv[:, :, :seq_len]

        # Back to tokens-first layout: (B, 384, 65) -> (B, 65, 384)
        x_conv = x_conv.transpose(1, 2)

        # SiLU ("Sigmoid Linear Unit", also called Swish): silu(u) = u * sigmoid(u),
        # where sigmoid(u) = 1 / (1 + e^-u) squashes u into (0, 1). So silu(u) is
        # about u for large positive u, about 0 for large negative u, and SMOOTH in
        # between (e.g. silu(-1) = -0.27, silu(0) = 0, silu(1) = 0.73).
        # Why not ReLU (max(0, u))? ReLU has a sharp corner at 0 and outputs exactly
        # zero, with zero gradient, for every negative input, so a unit that drifts
        # negative can stop learning entirely. SiLU lets a little signal and
        # gradient through for negative inputs. Mamba uses SiLU throughout.
        # (B, 65, 384) -> (B, 65, 384)
        x_conv = F.silu(x_conv)

        # ----- Algorithm 1, lines 8-9: per-token write and read gates -----

        # B: how strongly this token writes into each of the 16 state slots.
        # (B, 65, 384) -> (B, 65, 16)
        B = self.proj_B(x_conv)

        # C: how strongly each of the 16 slots is read out at this token.
        # (B, 65, 384) -> (B, 65, 16)
        C = self.proj_C(x_conv)

        # ----- Algorithm 1, line 11: Delta = softplus(Linear(x') + bias) -----

        # Raw per-token, per-channel step size; can be any real number.
        # (B, 65, 384) -> (B, 65, 384), then + delta_bias (384,) broadcast over
        # batch and tokens -> (B, 65, 384)
        delta_raw = self.proj_delta(x_conv) + self.delta_bias

        # softplus(u) = log(1 + exp(u)), a smooth version of ReLU that is ALWAYS
        # positive. Worked examples:
        #   u = -5:  log(1 + 0.0067)  = 0.0067   (tiny but still above zero)
        #   u =  0:  log(1 + 1)       = 0.6931
        #   u =  5:  log(1 + 148.41)  = 5.0067   (about u itself for large u)
        # Delta must be positive because it is a DURATION ("how long the tap is
        # open"); a negative duration would turn decay into growth.
        # (B, 65, 384) -> (B, 65, 384)
        delta = F.softplus(delta_raw)

        # ----- Algorithm 1, line 12: A = -exp(A_log), always negative -----
        # (384, 16) -> (384, 16)
        A = -torch.exp(self.A_log)

        # ----- Algorithm 1, lines 13-22: discretise and scan -----
        # (B, 65, 384) -> (B, 65, 384)
        y = selective_scan(x_conv, delta, A, B, C)

        # This direction's output, one 384-vector per token
        return y


class VimBlock(nn.Module):
    """One bidirectional Vim block: Algorithm 1, lines 1-28.

    Shapes (defaults): (B, 65, 192) -> (B, 65, 192)
    """

    def __init__(self, embed_dim=192, expand=2, d_state=16, conv_kernel=4,
                 dt_min=0.001, dt_max=0.1):
        # Parent constructor first, so nn.Module can register our layers
        super().__init__()

        # E = expand * D = 2 * 192 = 384
        expand_dim = expand * embed_dim
        # Remember both widths for forward()
        self.embed_dim = embed_dim
        self.expand_dim = expand_dim

        # Algorithm 1, line 2: Norm. LayerNorm rescales each token's 192 numbers to
        # mean 0 and variance 1, then applies a learned scale and shift (2 * 192
        # parameters). Normalising BEFORE the block's computation ("pre-norm")
        # keeps the inputs to every block on the same scale, however deep we are.
        self.norm = nn.LayerNorm(embed_dim)

        # Algorithm 1, lines 3-4: Linear^x and Linear^z, both D -> E.
        # We build them as ONE Linear D -> 2E (192 -> 768) and split its output in
        # forward(). This is exactly the same maths as two separate Linears: the
        # 768 x 192 weight matrix is just the two 384 x 192 matrices stacked on top
        # of each other, so the first 384 outputs ARE Linear^x and the last 384 ARE
        # Linear^z. One big matrix multiply is a little faster than two small ones.
        # bias=False follows the official Mamba code.
        self.in_proj = nn.Linear(embed_dim, 2 * expand_dim, bias=False)

        # Algorithm 1, line 6: one DirectionalSSM per direction, with SEPARATE
        # weights (conv1d, B/C/Delta projections, A). Two separate objects means
        # two separate sets of weights: nothing is shared between them. Assigning
        # each to self.<name> registers it with nn.Module, so the weights of both
        # appear in .parameters() and are trained.
        self.forward_ssm = DirectionalSSM(expand_dim, d_state, conv_kernel, dt_min,
                                          dt_max)
        self.backward_ssm = DirectionalSSM(expand_dim, d_state, conv_kernel, dt_min,
                                           dt_max)

        # Algorithm 1, line 28: Linear^T, E -> D (384 -> 192), back to model width
        self.out_proj = nn.Linear(expand_dim, embed_dim, bias=False)

    def forward(self, tokens):
        # tokens: T_{l-1}, shape (B, 65, 192)
        # Keep the untouched input for the residual connection at line 28
        residual = tokens

        # Algorithm 1, line 2: T' = Norm(T_{l-1}).  (B, 65, 192) -> (B, 65, 192)
        normed = self.norm(tokens)

        # Algorithm 1, lines 3-4, as one matrix multiply: (B, 65, 192) -> (B, 65, 768)
        xz = self.in_proj(normed)

        # ...then the explicit split along the last axis. `...` (Ellipsis) means
        # "all the leading dimensions, however many there are".
        # Line 3: x = the first E = 384 outputs.  (B, 65, 768) -> (B, 65, 384)
        x = xz[..., : self.expand_dim]
        # Line 4: z = the last E = 384 outputs.   (B, 65, 768) -> (B, 65, 384)
        z = xz[..., self.expand_dim :]

        # Each direction's scan output will be stored here under its name.
        # `{}` is an empty DICTIONARY (Java: `new HashMap<String, Tensor>()`);
        # `y["forward"] = ...` puts an entry in, `y["forward"]` reads it back.
        y = {}

        # Algorithm 1, line 6: for o in {forward, backward}.
        # Looping over a tuple of two strings runs the body twice, once with
        # direction = "forward" and once with direction = "backward".
        for direction in ("forward", "backward"):
            if direction == "forward":
                # Lines 7-22 with the FORWARD weights, reading tokens 0, 1, ..., 64
                # in order.  (B, 65, 384) -> (B, 65, 384)
                y[direction] = self.forward_ssm(x)
            else:
                # The backward scan reads tokens 64, 63, ..., 0. Rather than writing
                # a second, mirrored scan, we REVERSE the sequence, run the very
                # same forward-reading code (with the BACKWARD weights), and
                # reverse the result back so that row t again belongs to token t.
                # torch.flip(x, dims=[1]) reverses dimension 1 (the 65 tokens).
                # Note the conv is causal on the flipped sequence, so in the
                # backward direction it looks at the 3 tokens AFTER each token.
                # (B, 65, 384) -> flip -> scan -> flip -> (B, 65, 384)
                x_reversed = torch.flip(x, dims=[1])
                y_reversed = self.backward_ssm(x_reversed)
                y[direction] = torch.flip(y_reversed, dims=[1])

        # Algorithm 1, lines 25-26: gate each direction with SiLU(z).
        # z is a second copy of the (projected) input that acts as a VOLUME KNOB per
        # token and channel: silu(z) near 0 mutes that part of the scan's output,
        # larger values let it through. This is a minor architectural detail,
        # inherited from Mamba (it is a "gated" unit, like in GLU-style networks).
        # (B, 65, 384) * (B, 65, 384) -> (B, 65, 384), elementwise
        gate = F.silu(z)
        y_forward_gated = y["forward"] * gate
        y_backward_gated = y["backward"] * gate

        # Algorithm 1, line 28, first half: Linear^T(y'_forward + y'_backward).
        # Sum the two directions, (B, 65, 384) + (B, 65, 384) -> (B, 65, 384),
        # then project back down to the model width: (B, 65, 384) -> (B, 65, 192)
        out = self.out_proj(y_forward_gated + y_backward_gated)

        # Algorithm 1, line 28, second half: the residual connection, + T_{l-1}.
        # The block only ADDS its contribution to the running representation; it
        # never replaces it. If a block has nothing useful to say, it can output
        # roughly zero and the input passes through untouched. In the backward
        # pass, the "+" also gives gradients a direct path to earlier blocks.
        # Together this is what lets 12 (or the paper's 24) blocks stack without
        # the signal, or its gradient, dying out on the way.
        # (B, 65, 192) + (B, 65, 192) -> (B, 65, 192)
        return out + residual
