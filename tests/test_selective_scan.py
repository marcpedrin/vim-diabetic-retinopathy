"""Test selective_scan: shapes, behaviour at the two extremes, a hand-worked case,
and gradients.

What is checked:
    - shape: (Bt, M, E) in, (Bt, M, E) out
    - A very negative (instant forgetting): y_t depends only on x_t
    - A near zero (no forgetting): the state accumulates like a running sum
    - a 2-step example small enough to compute by hand, digit by digit
    - backward(): every input receives a finite, non-zero gradient, so the
      Linear layers that produce delta, B and C (and A itself) can be trained
"""

import math  # standard library maths (math.exp for the hand-computed case)
import os  # for building file paths
import sys  # for editing the module search path

# Add the project root to Python's search path so `import vim` works
# even when this file is run directly (python tests/test_selective_scan.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # tensors

from vim.selective_scan import selective_scan  # the function under test


def make_inputs(batch=2, seq_len=65, channels=384, state_dim=16, seed=0):
    """Random but valid inputs: delta > 0 and A < 0, as the Vim block guarantees."""
    # A private generator so this helper does not disturb the global random state
    g = torch.Generator().manual_seed(seed)
    # Input sequence
    x = torch.randn(batch, seq_len, channels, generator=g)
    # Step sizes in (0.01, 0.11): positive, like the softplus output in the block
    delta = torch.rand(batch, seq_len, channels, generator=g) * 0.1 + 0.01
    # Decay rates in (-1.5, -0.5): strictly negative, like -exp(A_log)
    A = -(torch.rand(channels, state_dim, generator=g) + 0.5)
    # Per-token write and read gates
    B = torch.randn(batch, seq_len, state_dim, generator=g)
    C = torch.randn(batch, seq_len, state_dim, generator=g)
    # PYTHON CONCEPT: returning several values at once returns a TUPLE, which the
    # caller can unpack: `x, delta, A, B, C = make_inputs()`.
    return x, delta, A, B, C


def test_shape():
    # Default-config sizes: 65 tokens, E = 384, N = 16
    x, delta, A, B, C = make_inputs()

    y = selective_scan(x, delta, A, B, C)

    # Print shapes so we can see them with `pytest -s` or when run directly
    print("x shape:", tuple(x.shape), " y shape:", tuple(y.shape))

    # One output vector per token, same shape as the input
    assert y.shape == (2, 65, 384)


def test_fast_decay_output_depends_only_on_current_token():
    # With A = -10000 and delta = 1, A_bar = exp(-10000) = 0 (to float precision):
    # the buckets are emptied completely at every step, so nothing from earlier
    # tokens survives. Then h_t = B_bar_t * x_t and y_t = sum_n C_tn * B_tn * x_t.
    x, _, _, B, C = make_inputs(batch=1, seq_len=10, channels=4, state_dim=3)
    delta = torch.ones(1, 10, 4)
    A = torch.full((4, 3), -10000.0)

    y = selective_scan(x, delta, A, B, C)

    # Change ONLY the first token's input, drastically
    x_changed = x.clone()
    x_changed[:, 0] = x_changed[:, 0] + 1000.0
    y_changed = selective_scan(x_changed, delta, A, B, C)

    # The output at token 0 changes, of course...
    assert not torch.allclose(y[:, 0], y_changed[:, 0])
    # ...but every later token is unaffected: the past was forgotten
    assert torch.allclose(y[:, 1:], y_changed[:, 1:])

    # And each output equals the "current token only" formula. (B * C).sum over the
    # N slots gives (1, 10); unsqueeze(-1) makes it (1, 10, 1) to broadcast over E.
    expected = x * (B * C).sum(dim=-1).unsqueeze(-1)
    assert torch.allclose(y, expected, atol=1e-5)


def test_no_decay_accumulates_like_running_sum():
    # With A = -1e-8, A_bar = exp(-1e-8) = 0.99999999, i.e. essentially 1: nothing
    # leaks. With delta = 1, B = 1 and a single state slot (N = 1),
    #   h_t = h_{t-1} + x_t    so   h_t = x_0 + x_1 + ... + x_t
    # and with C = 1, y_t = h_t: the running (cumulative) sum of the inputs.
    x = torch.randn(2, 20, 5)
    delta = torch.ones(2, 20, 5)
    A = torch.full((5, 1), -1e-8)
    B = torch.ones(2, 20, 1)
    C = torch.ones(2, 20, 1)

    y = selective_scan(x, delta, A, B, C)

    # torch.cumsum computes the running sum along the sequence dimension
    expected = torch.cumsum(x, dim=1)
    assert torch.allclose(y, expected, atol=1e-4)


def test_hand_computed_two_steps():
    # The smallest interesting case: 1 image, 2 tokens, 1 channel, 2 state slots.
    #
    #   x     = [1, 2]               (token 0 has value 1, token 1 has value 2)
    #   delta = [0.5, 1.0]
    #   A     = [-1, -2]             (slot 0 leaks slowly, slot 1 leaks fast)
    #   B     = token 0: [1,   0]    (write only into slot 0)
    #           token 1: [0.5, 1]
    #   C     = token 0: [1,   1]
    #           token 1: [2,  -1]
    #
    # Token 0 (h starts at [0, 0]):
    #   A_bar = exp(0.5 * [-1, -2]) = [e^-0.5, e^-1]   (irrelevant: h is still 0)
    #   B_bar = 0.5 * [1, 0]        = [0.5, 0]
    #   h     = A_bar * [0, 0] + B_bar * 1 = [0.5, 0]
    #   y_0   = 1 * 0.5 + 1 * 0     = 0.5
    #
    # Token 1:
    #   A_bar = exp(1.0 * [-1, -2]) = [e^-1, e^-2]
    #   B_bar = 1.0 * [0.5, 1]      = [0.5, 1]
    #   h     = [e^-1 * 0.5, e^-2 * 0] + [0.5 * 2, 1 * 2] = [0.5 e^-1 + 1, 2]
    #   y_1   = 2 * (0.5 e^-1 + 1) + (-1) * 2 = e^-1 + 2 - 2 = e^-1 = 0.36788
    x = torch.tensor([[[1.0], [2.0]]])  # (1, 2, 1)
    delta = torch.tensor([[[0.5], [1.0]]])  # (1, 2, 1)
    A = torch.tensor([[-1.0, -2.0]])  # (1, 2)
    B = torch.tensor([[[1.0, 0.0], [0.5, 1.0]]])  # (1, 2, 2)
    C = torch.tensor([[[1.0, 1.0], [2.0, -1.0]]])  # (1, 2, 2)

    y = selective_scan(x, delta, A, B, C)

    # The two numbers worked out above, in shape (1, 2, 1)
    expected = torch.tensor([[[0.5], [math.exp(-1.0)]]])
    print("y =", y.flatten().tolist(), " expected =", expected.flatten().tolist())
    assert torch.allclose(y, expected, atol=1e-6)


def test_gradients_reach_every_input():
    # Small sizes keep the test fast; the maths is the same at full size
    x, delta, A, B, C = make_inputs(batch=2, seq_len=8, channels=6, state_dim=4)

    # requires_grad_() asks autograd to compute d(output)/d(this tensor)
    for tensor in (x, delta, A, B, C):
        tensor.requires_grad_()

    # Any scalar works as a "loss" for checking that gradients flow
    y = selective_scan(x, delta, A, B, C)
    y.sum().backward()

    # Every input got a gradient that is finite (no NaN or inf) and not all zero
    for name, tensor in [("x", x), ("delta", delta), ("A", A), ("B", B), ("C", C)]:
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} has non-finite gradient"
        assert tensor.grad.abs().sum() > 0, f"{name} has an all-zero gradient"


# Allow running as a plain script: python tests/test_selective_scan.py
if __name__ == "__main__":
    test_shape()
    test_fast_decay_output_depends_only_on_current_token()
    test_no_decay_accumulates_like_running_sum()
    test_hand_computed_two_steps()
    test_gradients_reach_every_input()
    print("OK")
