"""Test VimBlock: shape, separate directional weights, direction-awareness,
a stable 12-block stack, and gradient flow.

What is checked:
    - shape: (2, 65, 192) -> (2, 65, 192)
    - the forward and backward branches own genuinely separate parameters
    - the block is direction-aware: reversing the input does NOT simply reverse
      the output (and it WOULD if both branches shared the same weights)
    - 12 stacked blocks neither blow up nor shrink the signal to nothing
    - gradients reach every parameter of a 12-block stack
    - A = -exp(A_log) is negative, and Delta starts inside [dt_min, dt_max]
"""

import os  # for building file paths
import sys  # for editing the module search path

# Add the project root to Python's search path so `import vim` works
# even when this file is run directly (python tests/test_vim_block.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # tensors
import torch.nn as nn  # nn.Sequential, used to stack blocks
import torch.nn.functional as F  # softplus, to check the initial Delta

from vim.vim_block import VimBlock  # the module under test


def test_vim_block_shape():
    # Fixed seed so the random weights (and so the test) are the same every run
    torch.manual_seed(0)
    # Default settings: D = 192, E = 384, N = 16, conv kernel 4
    block = VimBlock()

    # A batch of 2 sequences of 65 tokens (64 patches + class token)
    x = torch.randn(2, 65, 192)
    out = block(x)

    # Print shapes so we can see them with `pytest -s` or when run directly
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(out.shape))

    # Same shape out as in: blocks can be stacked
    assert out.shape == (2, 65, 192)


def test_directions_have_separate_parameters():
    torch.manual_seed(0)
    block = VimBlock()

    # dict(...) of named_parameters() maps names like "conv1d.weight" to tensors
    forward_params = dict(block.forward_ssm.named_parameters())
    backward_params = dict(block.backward_ssm.named_parameters())

    # Both branches have the same LIST of weights (same architecture)...
    assert forward_params.keys() == backward_params.keys()
    print("Per-direction parameters:", sorted(forward_params.keys()))

    for name in forward_params:
        # ...but every one is a DIFFERENT tensor in memory. `is not` compares object
        # identity (Java's `!=` on references); data_ptr() is the memory address of
        # the numbers, so this proves no storage is shared.
        assert forward_params[name] is not backward_params[name]
        assert forward_params[name].data_ptr() != backward_params[name].data_ptr()

    # And the randomly initialised weights hold different values in each branch
    assert not torch.equal(forward_params["conv1d.weight"],
                           backward_params["conv1d.weight"])
    assert not torch.equal(forward_params["proj_delta.weight"],
                           backward_params["proj_delta.weight"])


def test_block_is_direction_aware():
    torch.manual_seed(0)
    block = VimBlock()
    # eval() mode; this block has no dropout, but it is the right habit for tests
    block.eval()

    x = torch.randn(2, 65, 192)
    # A helper that reverses the token order (dimension 1)
    def reverse(t):
        return torch.flip(t, dims=[1])

    # with torch.no_grad(): we only need outputs here, no gradients
    with torch.no_grad():
        # Output for the input read in reverse, and the reversed normal output
        out_of_reversed = block(reverse(x))
        reversed_out = reverse(block(x))

    # If the block treated both directions identically, these would be equal: the
    # forward branch on the reversed input would do what the backward branch did
    # on the original. They differ, because each direction has its own weights.
    assert not torch.allclose(out_of_reversed, reversed_out, atol=1e-4)

    # Control experiment: copy the forward weights INTO the backward branch. Now
    # the block is symmetric, and reversing the input exactly reverses the output.
    # This proves the difference above really comes from the separate weights.
    block.backward_ssm.load_state_dict(
        block.forward_ssm.state_dict()
    )
    with torch.no_grad():
        out_of_reversed = block(reverse(x))
        reversed_out = reverse(block(x))
    assert torch.allclose(out_of_reversed, reversed_out, atol=1e-5)


def test_stack_of_12_blocks_stays_in_a_sane_range():
    torch.manual_seed(0)
    # nn.Sequential runs its modules one after another, like the model will.
    # PYTHON CONCEPT: a LIST COMPREHENSION `[expr for i in range(12)]` builds a
    # list by evaluating `expr` 12 times; here, 12 independent VimBlocks.
    # `*list` then UNPACKS the list into separate arguments: Sequential(b1,...,b12).
    # The `_` loop variable is the convention for "a counter we never use".
    stack = nn.Sequential(*[VimBlock() for _ in range(12)])
    stack.eval()

    # Input with standard deviation 1, like normalised embeddings
    x = torch.randn(2, 65, 192)

    # Scenario 1: fresh initialisation. With small initial step sizes each block
    # adds very little, so the stack starts close to the identity function.
    with torch.no_grad():
        out = stack(x)
    std = out.std().item()
    added = (out - x).std().item()
    print(f"Fresh init : output std {std:.3f}, part added by 12 blocks {added:.5f}")
    # No NaN / infinity anywhere, and neither exploded nor collapsed
    assert torch.isfinite(out).all()
    assert 0.5 < std < 10.0

    # Scenario 2: "awake" branches. Scenario 1 is easy because the blocks are
    # nearly silent, so also test with every Delta pushed up to about 1 (the
    # scans then forget and write strongly), which is closer to a trained model.
    # log(expm1(1.0)) is the inverse softplus of 1.0 (see vim/vim_block.py).
    awake_bias = torch.log(torch.expm1(torch.tensor(1.0)))
    with torch.no_grad():
        for block in stack:
            block.forward_ssm.delta_bias.fill_(awake_bias)
            block.backward_ssm.delta_bias.fill_(awake_bias)
        out = stack(x)
    std = out.std().item()
    added = (out - x).std().item()
    print(f"Awake Delta: output std {std:.3f}, part added by 12 blocks {added:.5f}")
    # Still finite and within a sane range: the residual keeps at least the
    # input's own scale, and the pre-norm keeps each block's addition modest.
    assert torch.isfinite(out).all()
    assert 0.5 < std < 10.0
    # And the blocks are now genuinely contributing
    assert added > 0.01


def test_gradient_flow_through_stack():
    torch.manual_seed(0)
    stack = nn.Sequential(*[VimBlock() for _ in range(12)])

    # Smaller batch and sequence to keep the test quick; same code path
    x = torch.randn(1, 17, 192, requires_grad=True)
    out = stack(x)
    out.sum().backward()

    # The input itself receives a gradient: the signal reaches all the way back
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # Every parameter of every block receives a finite, non-zero gradient,
    # including the FIRST block's (the hardest to reach) and both directions
    for name, p in stack.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum() > 0, f"{name} has an all-zero gradient"


def test_A_is_negative_and_delta_starts_in_range():
    torch.manual_seed(0)
    ssm = VimBlock().forward_ssm

    # A = -exp(A_log): every entry strictly negative, and at initialisation equal
    # to -1, -2, ..., -16 along the state axis (the S4D-real initialisation)
    A = -torch.exp(ssm.A_log)
    assert (A < 0).all()
    assert torch.allclose(A[0], -torch.arange(1, 17).float())

    # With a zero projection output, Delta = softplus(delta_bias) lies in
    # [dt_min, dt_max] = [0.001, 0.1] for every channel
    delta0 = F.softplus(ssm.delta_bias)
    assert (delta0 >= 0.001 - 1e-6).all() and (delta0 <= 0.1 + 1e-6).all()


# Allow running as a plain script: python tests/test_vim_block.py
if __name__ == "__main__":
    test_vim_block_shape()
    test_directions_have_separate_parameters()
    test_block_is_direction_aware()
    test_stack_of_12_blocks_stays_in_a_sane_range()
    test_gradient_flow_through_stack()
    test_A_is_negative_and_delta_starts_in_range()
    print("OK")
