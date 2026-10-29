"""Test that PatchEmbed turns (2, 3, 128, 128) images into (2, 64, 192) tokens."""

import os  
import sys  

# Add the project root to Python's search path so `import vim` works
# even when this file is run directly (python tests/test_patch_embed.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  
from vim.patch_embed import PatchEmbed  

def test_patch_embed_shape():
    model = PatchEmbed()
    # Fake img
    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    #test
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(out.shape))
    assert out.shape == (2, 64, 192)


# Allow running as a plain script: python tests/test_patch_embed.py
if __name__ == "__main__":
    test_patch_embed_shape()
    print("OK")
