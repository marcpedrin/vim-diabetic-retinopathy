"""selective_scan: the state space model recurrence at the heart of Mamba and Vim.

What this file does:
    Runs the discrete state space model (SSM) along a sequence of tokens, one
    token at a time, carrying a hidden state h from each token to the next,
    and returns one output vector per token.

Vim paper correspondence (arXiv 2401.09417, Section 3.1 and Algorithm 1):
    - Eq. 2 (Section 3.1), the discrete recurrence that this file computes:
          h_t = A_bar h_{t-1} + B_bar x_t
          y_t = C h_t
    - Eq. 1 (Section 3.1), the discretisation that produces A_bar and B_bar:
          A_bar = exp(Delta A)
          B_bar = (Delta A)^(-1) (exp(Delta A) - I) . Delta B
      We use exp(Delta A) exactly, and the SIMPLIFIED B_bar = Delta B (see
      "Deliberate differences" below).
    - Algorithm 1 (page 6), lines 13-14 and 16-22, transcribed:
          13: A_bar_o : (B, M, E, N) <- Delta_o (x) Parameter^A_o
          14: B_bar_o : (B, M, E, N) <- Delta_o (x) B_o
          15: /* initialise h_o and y_o with 0 */
          16: h_o : (B, E, N) <- zeros(B, E, N)
          17: y_o : (B, M, E) <- zeros(B, M, E)
          18: /* SSM recurrent */
          19: for i in {0, ..., M-1} do
          20:     h_o = A_bar_o[:, i, :, :] (.) h_o
                        + B_bar_o[:, i, :, :] (.) x'_o[:, i, :, None]
          21:     y_o[:, i, :] = h_o (x) C_o[:, i, :]
          22: end for
      ("(x)" is the paper's outer/broadcast product, "(.)" is elementwise.)
      Lines 1-12 and 23-28 (normalisation, projections, conv, gating, residual)
      live in vim/vim_block.py, which calls this function once per direction.

Shapes (defaults: batch Bt, M = 65 tokens, E = 384 channels, N = 16 state slots):
    x      (Bt, 65, 384)   input sequence, after conv1d + SiLU   [Alg. 1 line 7]
    delta  (Bt, 65, 384)   step size, per token and channel     [Alg. 1 line 11]
    A      (384, 16)       learned decay rates, always negative  [Alg. 1 line 12]
    B      (Bt, 65, 16)    per-token write gate                   [Alg. 1 line 8]
    C      (Bt, 65, 16)    per-token read gate                    [Alg. 1 line 9]
    y      (Bt, 65, 384)   output sequence

Deliberate differences from the paper:
    1. Algorithm 1 line 13 is PRINTED as "Delta (x) Parameter^A", without the
       exponential. Eq. 1 and the official Mamba/Vim code both apply exp(), and
       without it A_bar would be a negative number, which cannot be a "fraction
       of memory kept". We follow Eq. 1: A_bar = exp(Delta * A).
    2. B_bar = Delta * B is the simplified discretisation used by Algorithm 1
       line 14 AND by the official Mamba code, not the full zero-order-hold
       formula of Eq. 1. Why it is fine: for a small step Delta*A, the factor
       (exp(Delta A) - 1) / (Delta A) is approximately 1 (because
       exp(u) is about 1 + u for small u), so the full formula is approximately
       Delta * B anyway. The simple form is cheaper and is what the paper prints.
    3. Line 17 pre-allocates y with zeros and fills it row by row. We instead
       collect the rows in a Python list and stack them at the end (line 22).
       The result is identical; it avoids writing in place into a tensor that
       autograd is tracking, which is a common source of confusing errors.
    4. The scan is a plain Python `for` loop (see "Speed" below).

=====================================================================================
THE PICTURE TO KEEP IN YOUR HEAD: carrying a bucket of water along a row of patches
=====================================================================================
    Imagine the 65 tokens as 65 students sitting in one row. A note (the hidden
    state h) is passed from student to student. For each of the E = 384 channels
    the note has N = 16 small buckets of water, so h has shape (E, N) per image.

    At every student (token t), three things happen, in this order:
      1. LEAK.  Every bucket loses some water. A_bar (a number between 0 and 1)
                is the fraction that SURVIVES. A_bar near 1: long memory. A_bar
                near 0: the bucket is emptied and the past is forgotten.
      2. POUR.  This student pours their own value x_t into the buckets.
                B_bar decides how much goes into each of the 16 buckets.
      3. READ.  The output y_t is read off the buckets. C decides how much each
                of the 16 buckets counts towards the output (a weighted sum).

    Delta is the master dial: "how long is the tap open at this student?".
      - large Delta: A_bar = exp(Delta*A) is small, so the old water mostly
        leaks away, AND B_bar = Delta*B is large, so a lot of the new value
        pours in. "This token is important, reset and focus on it."
      - small Delta: A_bar is close to 1 and B_bar is small. "Barely notice
        this token, keep carrying what you had."

Why A is kept NEGATIVE:
    A_bar = exp(Delta * A). Delta is always positive (it is a duration). If A
    is negative, Delta * A is negative, and exp of a negative number lies in
    (0, 1). So A_bar is a genuine "fraction kept": the water level can only
    shrink between pours and the note never explodes. If A were positive,
    exp(Delta * A) > 1 would MULTIPLY the memory at every one of the 65 steps:
    1.1 ** 65 is already about 490, and it would grow without limit with longer
    sequences. vim/vim_block.py guarantees A < 0 by storing log(-A).

Why "selective": B, C and Delta are computed PER TOKEN (paper Section 3.1):
    In the older S4 model, A, B, C and Delta were fixed numbers, the same for
    every token. The bucket then leaks and fills in exactly the same way at
    every student, whatever they say, so the model cannot decide that THIS
    token is worth remembering and THAT one is noise. Mamba makes B, C and
    Delta depend on the token itself (they are computed from x by small Linear
    layers in vim/vim_block.py). That input-dependence is what "selective"
    means: each token chooses how much to write, what to read, and how much of
    the past to keep. That is why B, C and Delta carry a token dimension M
    here, while A (the base decay rates) does not.

What is NOT coded, and why:
    - The continuous-time system h'(t) = A h(t) + B x(t), y(t) = C h(t)
      (printed in Section 3.1 without an equation number) is the theory the
      discrete model comes from. Computers work in steps, so it is never
      coded; Eq. 1 turns it into the step-by-step form of Eq. 2.
    - Eq. 3 rewrites the recurrence as ONE big convolution with a fixed kernel
      K_bar = (C B_bar, C A_bar B_bar, ..., C A_bar^(M-1) B_bar). That trick
      only works when A_bar, B_bar and C are the same at every step (as in
      S4). In a SELECTIVE model they change at every token, so no single fixed
      kernel exists and Eq. 3 does not apply. We must run the recurrence.

Speed (why this is slow, and why we keep it that way):
    The loop runs M = 65 steps one after another, and each step needs the
    result of the previous one, so its cost grows linearly with the sequence
    length, O(M), and it cannot be spread over the GPU's thousands of cores
    along the sequence. The paper (Section 3.5, "hardware-aware" design, from
    Mamba) solves this with a fused CUDA kernel that runs a parallel scan and
    keeps h in fast on-chip memory. That is deliberately OUT OF SCOPE: the task
    forbids ready-made kernels, and the readable loop below is the actual
    mathematics, line for line. Trade-off chosen: explainable over fast.
"""

import torch  # main PyTorch library (tensors)


def selective_scan(x, delta, A, B, C):
    """Run the selective SSM recurrence (Eq. 2) along the sequence dimension.

    Shapes (defaults in brackets; Bt = batch size):
        x      (Bt, M, E)   [Bt, 65, 384]  input sequence after conv + SiLU
        delta  (Bt, M, E)   [Bt, 65, 384]  per-token, per-channel step size (> 0)
        A      (E, N)       [384, 16]      learned decay matrix, always negative
        B      (Bt, M, N)   [Bt, 65, 16]   per-token write gate
        C      (Bt, M, N)   [Bt, 65, 16]   per-token read gate
    Returns:
        y      (Bt, M, E)   [Bt, 65, 384]  output sequence
    """
    # ---------------------------------------------------------------------------------
    # Step 0: read the sizes off the inputs and check that they agree
    # ---------------------------------------------------------------------------------

    # PYTHON CONCEPT: tuple unpacking. `x.shape` is a tuple of 3 sizes, e.g.
    # (2, 65, 384), and `a, b, c = some_tuple` assigns its elements to a, b, c in
    # one go. So batch_size = 2, seq_len = 65, channels = 384.
    batch_size, seq_len, channels = x.shape
    # A tells us the number of state slots per channel: N = 16
    state_dim = A.shape[1]

    # Shape checks. Mistakes here would otherwise surface as a confusing error deep
    # inside the broadcasting below, or worse, silently produce the wrong result.
    assert delta.shape == x.shape, "delta must have the same shape as x: (Bt, M, E)"
    assert A.shape == (channels, state_dim), "A must have shape (E, N)"
    assert B.shape == (batch_size, seq_len, state_dim), "B must have shape (Bt, M, N)"
    assert C.shape == (batch_size, seq_len, state_dim), "C must have shape (Bt, M, N)"

    # ---------------------------------------------------------------------------------
    # Step 1: discretise, for ALL tokens at once   [Algorithm 1, lines 13-14]
    # ---------------------------------------------------------------------------------

    # PYTORCH CONCEPT: unsqueeze. `t.unsqueeze(d)` inserts a new dimension of size 1
    # at position d, without changing the data. `-1` means "at the very end".
    # We do this so that two tensors line up for broadcasting: a size-1 dimension is
    # stretched to match the other tensor's size in that dimension.

    # delta gets a trailing size-1 "state slot" dimension:
    #   delta (Bt, 65, 384) -> delta.unsqueeze(-1) (Bt, 65, 384, 1)
    # A (384, 16) is lined up from the RIGHT, as if it were (1, 1, 384, 16).
    # Broadcasting then gives every (image, token, channel) its own 16 numbers:
    #   (Bt, 65, 384, 1) * (384, 16) -> (Bt, 65, 384, 16)
    # Each entry is Delta[b, t, e] * A[e, n]: channel e's step size times the base
    # decay rate of its slot n.
    delta_A = delta.unsqueeze(-1) * A

    # Algorithm 1, line 13 (with the exp of Eq. 1): A_bar = exp(Delta * A).
    # Delta > 0 and A < 0, so Delta * A < 0, and exp of it lies strictly in (0, 1):
    # the fraction of each bucket that survives this step.  (Bt, 65, 384, 16)
    A_bar = torch.exp(delta_A)

    # Algorithm 1, line 14: B_bar = Delta * B (the simplified discretisation).
    # B is per token but NOT per channel: (Bt, 65, 16). We insert a channel axis at
    # position 2 so it lines up with delta's channel axis:
    #   B (Bt, 65, 16) -> B.unsqueeze(2) (Bt, 65, 1, 16)
    #   delta.unsqueeze(-1)              (Bt, 65, 384, 1)
    # Product: (Bt, 65, 384, 1) * (Bt, 65, 1, 16) -> (Bt, 65, 384, 16)
    # Meaning: at token t, all 384 channels share the same "which buckets to pour
    # into" pattern B[t], but each channel scales it by its own tap time Delta.
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)

    # Memory note: A_bar and B_bar hold Bt * 65 * 384 * 16 numbers each, about
    # 51 MB per tensor for a batch of 32 in float32. Algorithm 1 computes them up
    # front in the same way; the official kernel avoids storing them at all.

    # ---------------------------------------------------------------------------------
    # Step 2: start with empty buckets   [Algorithm 1, lines 15-17]
    # ---------------------------------------------------------------------------------

    # Algorithm 1, line 16: h = zeros(Bt, E, N). One blank note per image, with
    # 16 empty buckets for each of the 384 channels: (Bt, 384, 16).
    # device=x.device puts it on the same device (CPU or GPU) as the inputs;
    # PyTorch refuses to mix devices in one operation. dtype=A_bar.dtype matches the
    # number format (e.g. float32), so the recurrence runs at A_bar's precision.
    h = torch.zeros(batch_size, channels, state_dim, device=x.device, dtype=A_bar.dtype)

    # Algorithm 1, line 17 pre-allocates y = zeros(Bt, M, E). We instead start an
    # empty Python LIST (Java: `new ArrayList<Tensor>()`) and append one row per
    # step; torch.stack turns the list into a tensor at the end (line 22).
    ys = []

    # ---------------------------------------------------------------------------------
    # Step 3: walk along the row of tokens   [Algorithm 1, lines 18-22]
    # ---------------------------------------------------------------------------------

    # Algorithm 1, line 19: for i in {0, ..., M-1}.
    # PYTHON CONCEPT: `for t in range(seq_len):` is Java's
    # `for (int t = 0; t < seq_len; t++)`. range(65) produces 0, 1, ..., 64.
    # This loop is SEQUENTIAL: step t needs the h produced by step t-1.
    for t in range(seq_len):
        # Pick out token t. Indexing a dimension with a single number REMOVES it:
        #   A_bar[:, t]  (Bt, 65, 384, 16) -> (Bt, 384, 16)   survival fractions
        #   B_bar[:, t]  (Bt, 65, 384, 16) -> (Bt, 384, 16)   pour amounts
        #   x[:, t]      (Bt, 65, 384)     -> (Bt, 384)       this token's values
        #   C[:, t]      (Bt, 65, 16)      -> (Bt, 16)        this token's read gate

        # The token's value must be poured into all 16 buckets of its channel, so
        # give it a trailing slot axis to broadcast over:
        #   x[:, t] (Bt, 384) -> x_t (Bt, 384, 1)
        x_t = x[:, t].unsqueeze(-1)

        # Algorithm 1, line 20: h = A_bar[:, i] (.) h + B_bar[:, i] (.) x[:, i, None]
        # This is Eq. 2's h_t = A_bar h_{t-1} + B_bar x_t.
        #   A_bar[:, t] * h      (Bt, 384, 16) * (Bt, 384, 16) -> (Bt, 384, 16)  LEAK
        #   B_bar[:, t] * x_t    (Bt, 384, 16) * (Bt, 384, 1)  -> (Bt, 384, 16)  POUR
        # When both shapes are equal, `*` is a plain elementwise product: every
        # bucket (b, e, n) is multiplied by its own survival fraction. In the second
        # product, x_t's size-1 slot axis is broadcast, so channel e's value is
        # poured into all 16 of its buckets, in the proportions given by B_bar.
        # Note that A_bar is DIAGONAL in the paper's matrix language: bucket n only
        # ever leaks into itself, never into bucket m. That is why an elementwise
        # product is enough and no matrix multiplication is needed.
        h = A_bar[:, t] * h + B_bar[:, t] * x_t

        # Algorithm 1, line 21: y[:, i] = h (x) C[:, i]. This is Eq. 2's y_t = C h_t.
        # C is shared by all channels, so give it a channel axis to broadcast over:
        #   C[:, t] (Bt, 16) -> C_t (Bt, 1, 16)
        C_t = C[:, t].unsqueeze(1)

        # READ: weight each bucket by C, then add up the 16 buckets of each channel:
        #   h * C_t          (Bt, 384, 16) * (Bt, 1, 16) -> (Bt, 384, 16)
        #   .sum(dim=-1)     (Bt, 384, 16) -> (Bt, 384)
        # `.sum(dim=-1)` adds along the LAST axis (the N state slots) and removes
        # it, collapsing 16 bucket readings into ONE output number per channel.
        # This is a dot product between h[b, e, :] and C[b, t, :] for every b and e.
        y_t = (h * C_t).sum(dim=-1)

        # Keep this token's output row; `append` adds to the end of the list
        ys.append(y_t)

    # Algorithm 1, line 22: end for. Turn the list of 65 tensors of shape (Bt, 384)
    # into one tensor, inserting a NEW sequence axis at position 1:
    #   65 x (Bt, 384) -> (Bt, 65, 384)
    y = torch.stack(ys, dim=1)

    # One output vector per token, same shape as the input x
    return y
