"""Vision Mamba (Vim) implemented from scratch in plain PyTorch.

This file runs the first time anything is imported from the `vim` package. The
imports below make the main pieces available directly from the package, so
other code can write `from vim import VisionMamba` instead of the longer
`from vim.vision_mamba import VisionMamba`.
"""

# Section 3.2: image -> patch tokens (the x_p W term of Eq. 4)
from .patch_embed import PatchEmbed

# Section 3.2: class token + position embeddings (the rest of Eq. 4)
from .tokens import ClassTokenPosEmbed

# Section 3.1, Eq. 2 / Algorithm 1 lines 13-22: the selective scan
from .selective_scan import selective_scan

# Section 3.3 / Algorithm 1: one bidirectional Vim block
from .vim_block import DirectionalSSM, VimBlock

# Sections 3.2 and 3.4: the full classifier, its builder and a parameter counter
from .vision_mamba import VisionMamba, build_model, count_parameters
