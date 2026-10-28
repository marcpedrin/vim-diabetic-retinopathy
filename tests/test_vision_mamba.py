"""Test VisionMamba: shape, parameter count, one learning step, and eval mode.

What is checked:
    - (2, 3, 128, 128) images give (2, 5) logits
    - the parameter count is printed, and is the exact number we expect
    - one AdamW step on a fixed random batch lowers the loss on that batch,
      i.e. gradients flow end to end and point in a useful direction
    - the model runs in eval mode under torch.no_grad(), and is deterministic
      there (same input -> same output)
    - the config builder produces the same architecture as the defaults
"""

import os  # for building file paths
import sys  # for editing the module search path

# Add the project root to Python's search path so `import vim` works
# even when this file is run directly (python tests/test_vision_mamba.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # tensors
import torch.nn.functional as F  # cross_entropy loss

from vim import VisionMamba, build_model, count_parameters  # under test
from vim.config import load_config  # to test the config-driven builder

# Path of the default config, relative to this test file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def test_vision_mamba_shape():
    torch.manual_seed(0)
    # Default model: 128 px, 12 blocks, middle class token, 5 classes
    model = VisionMamba()

    # 2 random "images"
    images = torch.randn(2, 3, 128, 128)
    logits = model(images)

    # Print shapes so we can see them with `pytest -s` or when run directly
    print("Input shape :", tuple(images.shape))
    print("Output shape:", tuple(logits.shape))

    # One score per DR grade for each of the 2 images
    assert logits.shape == (2, 5)


def test_parameter_count():
    model = VisionMamba()
    total = count_parameters(model)
    # The `:,` format adds thousands separators: 6857093 -> "6,857,093"
    print(f"Parameters: {total:,}  (paper Vim-Ti: about 7M with 24 blocks)")

    # The exact count for the default config; the arithmetic is in the docstring of
    # count_parameters (vim/vision_mamba.py). If this changes, the architecture did.
    assert total == 6_857_093


def test_one_optimiser_step_reduces_loss():
    torch.manual_seed(0)
    # A 2-block model keeps the test fast; the code path is identical to 12 blocks
    model = VisionMamba(depth=2)
    model.train()

    # A fixed batch of 4 random images with fixed labels
    images = torch.randn(4, 3, 128, 128)
    labels = torch.tensor([0, 1, 2, 3])

    # The optimiser used in training (paper Section 4.1 values)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    # Loss before the step
    loss_before = F.cross_entropy(model(images), labels)

    # The five-step training loop (explained in detail in train/engine.py)
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    # Loss on the SAME batch after one step
    with torch.no_grad():
        loss_after = F.cross_entropy(model(images), labels)

    print(f"Loss before: {loss_before.item():.4f}  after: {loss_after.item():.4f}")
    # One step downhill on this exact batch must lower its loss
    assert loss_after.item() < loss_before.item()


def test_eval_mode_under_no_grad():
    torch.manual_seed(0)
    model = VisionMamba(depth=2)
    # eval() switches off training-only behaviour (none in this model yet, but
    # always do it before evaluating)
    model.eval()

    images = torch.randn(2, 3, 128, 128)
    # torch.no_grad(): no computation graph is recorded, saving memory
    with torch.no_grad():
        out1 = model(images)
        out2 = model(images)

    # The output carries no gradient history...
    assert not out1.requires_grad
    # ...and is deterministic: the same images give exactly the same scores
    assert torch.equal(out1, out2)
    assert out1.shape == (2, 5)


def test_build_model_from_config():
    # The config-driven builder must give the same architecture as the defaults
    cfg = load_config(CONFIG_PATH)
    model = build_model(cfg.model)

    assert len(model.blocks) == cfg.model.depth
    assert model.cls_index == 32
    assert count_parameters(model) == count_parameters(VisionMamba())


# Allow running as a plain script: python tests/test_vision_mamba.py
if __name__ == "__main__":
    test_vision_mamba_shape()
    test_parameter_count()
    test_one_optimiser_step_reduces_loss()
    test_eval_mode_under_no_grad()
    test_build_model_from_config()
    print("OK")
