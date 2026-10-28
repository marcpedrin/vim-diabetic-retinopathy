"""Test ClassTokenPosEmbed: shapes, class token position, and registered parameters.

What is checked:
    - (2, 64, 192) patch tokens become a (2, 65, 192) sequence
    - the class token really lands at index 32 (the middle), with the patches
      shifted around it in their original order
    - the class token and position embeddings appear in `.parameters()`, which
      is what lets the optimiser train them
"""

import os  # for building file paths
import sys  # for editing the module search path

# Add the project root to Python's search path so `import vim` works
# even when this file is run directly (python tests/test_tokens.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # tensors

from vim.tokens import ClassTokenPosEmbed  # the module under test


def test_tokens_shape():
    # Default settings: 64 patches, D = 192, class token at index 32
    module = ClassTokenPosEmbed()

    # Fake output of PatchEmbed for a batch of 2 images
    x = torch.randn(2, 64, 192)

    # Insert the class token and add position embeddings
    out = module(x)

    # Print shapes so we can see them with `pytest -s` or when run directly
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(out.shape))

    # 2 images, 64 patches + 1 class token, 192 numbers per token
    assert out.shape == (2, 65, 192)


def test_class_token_lands_at_index_32():
    module = ClassTokenPosEmbed()

    # Make the result easy to read by hand:
    #   - position embeddings all zero, so they add nothing
    #   - class token all 100, so it is unmistakable
    # `with torch.no_grad():` tells PyTorch not to record these edits for
    # gradient computation; we are setting up the test, not training.
    with torch.no_grad():
        module.pos_embed.zero_()
        module.cls_token.fill_(100.0)

    # Patch tokens whose value is their own patch number: patch j is all j.
    # torch.arange(64) = [0, 1, ..., 63]; .float() makes them decimals.
    # .view(1, 64, 1) reshapes to (1, 64, 1); .expand(2, 64, 192) stretches it
    # (2 images, every one of the 192 numbers of patch j equal to j).
    x = torch.arange(64).float().view(1, 64, 1).expand(2, 64, 192)

    out = module(x)

    # Index 32 is the class token
    assert torch.all(out[:, 32] == 100.0)
    # Indices 0..31 are patches 0..31, unchanged
    assert torch.all(out[:, 31] == 31.0)
    assert torch.all(out[:, 0] == 0.0)
    # Index 33 is patch 32: everything after the class token shifted right by one
    assert torch.all(out[:, 33] == 32.0)
    # The last index (64) is the last patch (63)
    assert torch.all(out[:, 64] == 63.0)


def test_head_position_puts_class_token_first():
    # The ViT-style "head" variant from paper Table 5, for comparison
    module = ClassTokenPosEmbed(cls_index=0)
    with torch.no_grad():
        module.pos_embed.zero_()
        module.cls_token.fill_(100.0)

    out = module(torch.zeros(1, 64, 192))

    # The class token is now at the front
    assert torch.all(out[:, 0] == 100.0)


def test_parameters_are_registered():
    module = ClassTokenPosEmbed()

    # named_parameters() lists (name, tensor) pairs for every learnable weight.
    # A DICT COMPREHENSION `{k: v for k, v in pairs}` builds a dictionary in one
    # expression: here, from parameter name to its shape.
    shapes = {name: tuple(p.shape) for name, p in module.named_parameters()}
    print("Parameters:", shapes)

    # Both tables are learnable, with the shapes from Eq. 4
    assert shapes["cls_token"] == (1, 1, 192)
    assert shapes["pos_embed"] == (1, 65, 192)

    # And both have requires_grad=True, so gradients will be computed for them
    for p in module.parameters():
        assert p.requires_grad

    # Small random initialisation: not all zeros, and within the truncation bound
    assert module.pos_embed.abs().max() > 0
    assert module.pos_embed.abs().max() <= 0.04 + 1e-6


# Allow running as a plain script: python tests/test_tokens.py
if __name__ == "__main__":
    test_tokens_shape()
    test_class_token_lands_at_index_32()
    test_head_position_puts_class_token_first()
    test_parameters_are_registered()
    print("OK")
