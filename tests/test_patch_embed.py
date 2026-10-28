"""Test that PatchEmbed turns (2, 3, 128, 128) images into (2, 64, 192) tokens."""

import os  # for building file paths
import sys  # for editing the module search path

# Add the project root to Python's search path so `import vim` works
# even when this file is run directly (python tests/test_patch_embed.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # tensors

from vim.patch_embed import PatchEmbed  # the module under test


def test_patch_embed_shape():
    # Build the module with the default settings (128 image, 16 patch, D=192)
    model = PatchEmbed()

    # Fake batch of 2 random RGB images of 128x128 pixels
    x = torch.randn(2, 3, 128, 128)

    # Run the images through PatchEmbed
    out = model(x)

    # Print shapes so we can see them with `pytest -s` or when run directly
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(out.shape))

    # 2 images, 64 patches each, 192 numbers per patch
    assert out.shape == (2, 64, 192)


# Allow running as a plain script: python tests/test_patch_embed.py
if __name__ == "__main__":
    test_patch_embed_shape()
    print("OK")
