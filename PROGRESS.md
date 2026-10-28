# Progress log

One entry per phase: what was built, and what its tests prove.

## Phase 0: PatchEmbed (done before this log started)

- `vim/patch_embed.py`: cuts a (B, 3, 128, 128) image into 64 patches of 16x16 and
  projects each to D = 192 numbers with a Conv2d. This is the `x_p W` term of
  paper Eq. 4 (Section 3.2).
- `tests/test_patch_embed.py` proves the output shape is (B, 64, 192).

## Phase 1: repo scaffolding

Built:
- `configs/default.yaml`: every hyperparameter, each with the paper's value and the
  reason ours differs (128 px images instead of 224, 12 blocks instead of 24,
  30 epochs instead of 300, batch 32 instead of 1024).
- `vim/config.py`: loads the YAML into dataclasses (`Config`, `ModelConfig`, ...),
  computes derived values once (J = 64 patches, M = 65 tokens, E = 384, class
  token index 32), and supports `section.key=value` overrides from the command line.
- `train/utils.py`: `set_seed()` (Python, NumPy, torch CPU and CUDA, cuDNN
  deterministic), `seed_worker()` for DataLoader workers, `AverageMeter`,
  `save_checkpoint()` / `load_checkpoint()`, and a CSV `Logger`.
- `requirements.txt` and `.gitignore` (datasets, outputs, checkpoints, figures,
  LaTeX by-products, the personal reading guide).

Tests (`tests/test_config.py`) prove:
- the YAML loads and has the approved values;
- the derived values are right (64 patches, 65 tokens, E = 384, cls index 32);
- overrides change the right value with the right type, and typos are rejected;
- a config survives the round trip to a plain dictionary and back (as it must,
  because checkpoints store it that way).

Note on the paper's numbering (checked against arXiv 2401.09417v3): the
continuous-time system h'(t) = A h(t) + B x(t) is printed WITHOUT an equation
number. Eq. 1 is the zero-order-hold discretisation, Eq. 2 the discrete
recurrence, Eq. 3 the global convolution, Eq. 4 the token sequence T_0. In
Algorithm 1, line 17 initialises y, the inner loop is line 19, the state update
is line 20 and the read-out is line 21. All files use this numbering.

## Phase 2: tokens and position embeddings

Built:
- `vim/tokens.py`: `ClassTokenPosEmbed`, the rest of Eq. 4 (Section 3.2). A learnable
  class token `t_cls` (1, 1, 192) is inserted at the middle index 32 of the 64 patch
  tokens, then learnable position embeddings `E_pos` (1, 65, 192) are added:
  (B, 64, 192) -> (B, 65, 192). Both are initialised with a truncated normal,
  std 0.02, cut off at +-2 std (PyTorch's default cut-off of +-2 is absolute and
  would truncate nothing at this std, so it is passed explicitly).
- Comments cover why a class token exists, why the middle beats the front
  (Table 5: 76.1% vs 75.2%), why position embeddings are needed for a sequential
  scan, nn.Module / nn.Parameter / super().__init__() / forward(), and broadcasting.

Tests (`tests/test_tokens.py`) prove:
- the shape (2, 64, 192) -> (2, 65, 192);
- the class token lands exactly at index 32, with patches 0..31 before it and
  patches 32..63 shifted to indices 33..64 in their original order;
- the "head" variant puts it at index 0;
- both tables are registered learnable parameters with the Eq. 4 shapes, and the
  initial values are small, non-zero and within the truncation bound.

## Phase 3: the selective scan

Built:
- `vim/selective_scan.py`: one function, `selective_scan(x, delta, A, B, C) -> y`,
  implementing Eq. 2 (h_t = A_bar h_{t-1} + B_bar x_t, y_t = C h_t) as an explicit
  Python loop over the 65 tokens, i.e. Algorithm 1 lines 13-14 and 16-22.
  A_bar = exp(Delta * A) (Eq. 1; Algorithm 1 line 13 is printed without the exp),
  B_bar = Delta * B (the simplified discretisation used by Algorithm 1 line 14 and
  the official Mamba code). Shapes: x, delta (Bt, 65, 384); A (384, 16);
  B, C (Bt, 65, 16); y (Bt, 65, 384).
- Comments cover the bucket-of-water picture, why A < 0, what "selective" means,
  every unsqueeze/broadcast with its shapes, why the continuous form and Eq. 3
  (global convolution) are not coded, and why the slow O(M) loop is kept
  (the hardware-aware scan of Section 3.5 is out of scope).

Tests (`tests/test_selective_scan.py`) prove:
- the output shape equals the input shape (2, 65, 384);
- with A very negative (A_bar = 0) each output depends only on its own token,
  and equals x_t * sum_n(B_tn C_tn) exactly;
- with A near zero (A_bar = 1) the state is a running sum (matches torch.cumsum);
- a 2-token, 2-slot example worked out by hand gives y = [0.5, e^-1];
- backward() gives finite, non-zero gradients to x, delta, A, B and C.

## Phase 4: the bidirectional Vim block

Built:
- `vim/vim_block.py`: `DirectionalSSM` (Algorithm 1 lines 7-22 for one direction:
  depthwise causal Conv1d(k=4) + SiLU, Linear E->N for B and C, Linear E->E plus a
  Parameter bias + softplus for Delta, A = -exp(A_log) with the S4D-real
  initialisation, then `selective_scan`) and `VimBlock` (the whole of Algorithm 1:
  LayerNorm, one Linear D->2E split explicitly into x and z, a
  `for direction in ("forward", "backward")` loop with separate weights per
  direction, flip-scan-flip for the backward direction, SiLU(z) gating, sum,
  Linear E->D, residual). The module docstring transcribes Algorithm 1 line by line
  and maps each line to the code.
- Deviations documented: full E->E Delta projection as printed (official code is
  low-rank), LayerNorm (official Vim-Ti uses RMSNorm), no D*x skip term and no
  divide-by-2 (both in the official code, neither in the printed algorithm).

Tests (`tests/test_vim_block.py`) prove:
- shape (2, 65, 192) -> (2, 65, 192);
- the two directions own separate tensors (different memory, different values);
- the block is direction-aware: block(reverse(x)) != reverse(block(x)), and a
  control experiment shows that copying the forward weights into the backward
  branch makes the two equal, so the difference comes from the separate weights;
- a 12-block stack stays finite with output std ~1.0, both at initialisation and
  with every Delta pushed to ~1 ("awake" branches);
- every parameter of a 12-block stack gets a finite, non-zero gradient;
- A < 0 everywhere, equal to -1..-16 at initialisation, and the initial Delta
  lies in [0.001, 0.1].

Finding: at initialisation each block adds very little (std ~4e-4 for a unit-std
input; ~0.0015 summed over 12 blocks), so a fresh stack is close to the identity.
This comes from the small initial Delta and the absence of the D*x term. It is
harmless with AdamW, whose per-parameter step size does not depend on the raw
gradient scale.

## Phase 5: the full model

Built:
- `vim/vision_mamba.py`: `VisionMamba` = PatchEmbed -> ClassTokenPosEmbed -> 12 x
  VimBlock -> LayerNorm -> class token at index 32 -> Linear 192 -> 5. The five
  steps are commented against Section 3.2 (Eq. 4, T_l = Vim(T_{l-1}) + T_{l-1},
  f = Norm(T_L^0), p_hat = MLP(f)). Also `build_model(cfg.model)` and
  `count_parameters()`; `python -m vim.vision_mamba` prints a parameter breakdown.
- `vim/__init__.py` exports every public piece.

Parameter count: 6,857,093, versus the paper's ~7M for Vim-Ti. It is NOT half the
paper's size despite half the depth, because the printed Algorithm 1 line 11 uses a
full E->E Delta projection (147,456 weights per direction per block) where the
official code uses a rank-12 factorisation. Our block: 557,952 parameters; an
official block: ~282,000. The two Delta projections are 53% of each block.

Tests (`tests/test_vision_mamba.py`) prove:
- (2, 3, 128, 128) -> (2, 5);
- the parameter count is exactly 6,857,093 (printed);
- one AdamW step on a fixed batch lowers that batch's loss (1.81 -> 1.52);
- eval mode under torch.no_grad() works, records no gradient history, and is
  deterministic;
- `build_model` from the YAML config gives the same architecture as the defaults.
