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
