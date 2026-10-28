"""ClassTokenPosEmbed: inserts the class token and adds position embeddings.

What this file does:
    Takes the 64 patch tokens produced by PatchEmbed, inserts one extra
    learnable "class token" into the MIDDLE of the sequence (position 32), and
    adds a learnable position embedding to every one of the 65 tokens.

Vim paper correspondence (arXiv 2401.09417, Section 3.2, Eq. 4):
    Eq. 4 builds the first token sequence T_0 that enters the Vim blocks:

        T_0 = [ t_p^1 W ; ... ; t_cls ; ... ; t_p^J W ] + E_pos

    (The printed Eq. 4 shows t_cls at the front, the generic ViT form. The
    paper's ablation in Table 5 then shows that putting it in the MIDDLE works
    best, and the middle is the paper's default model. We follow the middle.)

    vim/patch_embed.py already produced the `t_p^j W` terms. This file adds
    the remaining two ingredients:
        t_cls  the class token, a learnable vector of D = 192 numbers
        E_pos  the position embeddings, one learnable D-vector per position,
               shape (J + 1, D) = (65, 192)

Shapes (defaults): patch tokens (B, 64, 192) -> token sequence (B, 65, 192)

Deliberate differences from the paper:
    None in the maths. Truncated-normal initialisation with std 0.02 follows
    the official Vim code (inherited from ViT/DeiT); we additionally truncate
    at 2 standard deviations, which the official code does not (its cut-off is
    effectively absent, see _init_weights). This only affects the starting point.

Why a class token exists at all:
    The model must output ONE prediction per image, but the blocks output 65
    vectors. Which one do we classify from? We add an extra token that belongs
    to no patch. It starts as the same learned vector for every image, and as it
    passes through the 12 blocks it collects information from all the patches.
    At the end, only this token is fed to the classifier (vim/vision_mamba.py).
    Think of it as an empty notebook passed round the class: everyone writes in
    it, and the teacher marks only the notebook.

Why the MIDDLE rather than the front (paper Table 5):
    head class token 75.2% top-1  vs  middle class token 76.1% top-1
    A Mamba scan reads the sequence in ONE direction, and its memory h fades
    with distance (each step multiplies the old memory by a number below 1).
    A token at the front is the FIRST thing the forward scan sees, before any
    patch, so the forward scan has nothing to tell it; only the backward scan
    reaches it, and the patches it hears most clearly are the last few it saw.
    A token in the middle is reached by BOTH scans, and the furthest patch is
    only 32 steps away instead of 64. Shortest average distance from every
    patch -> the least faded information -> the best accuracy.

Why position embeddings are needed even though the scan is sequential:
    It is true that a scan processes tokens in order, so it does have some
    sense of "before" and "after". But (a) once the 8x8 grid is flattened into
    a row, the scan cannot tell that patch 0 and patch 8 were vertical
    neighbours in the image, and (b) the per-token dials B, C and Delta are
    computed from each token on its own, so without a position signal a patch
    in the centre of the retina and one at the edge with identical pixels would
    get identical dials. E_pos gives every position its own learned "address",
    added to the token, so the model can learn what each location means.
"""

import torch  # main PyTorch library (tensors)
import torch.nn as nn  # neural network building blocks (Module, Parameter, ...)


# -------------------------------------------------------------------------------------
# PYTHON / PYTORCH CONCEPT: inheritance, nn.Module, super().__init__(), forward()
#
# `class ClassTokenPosEmbed(nn.Module):` means "ClassTokenPosEmbed EXTENDS
# nn.Module", exactly like Java's `class ClassTokenPosEmbed extends Module`. The
# name in brackets is the parent (base) class.
#
# nn.Module is PyTorch's base class for every layer and every model. Inheriting from
# it gives us, for free: automatic tracking of all learnable weights (so the
# optimiser can find them via `.parameters()`), moving them all to the GPU with
# `.to("cuda")`, saving them with `.state_dict()`, and switching train/eval mode.
#
# `super().__init__()` calls the PARENT's constructor (Java: `super();`). It must
# come first: nn.Module's constructor sets up the internal bookkeeping that the
# weight tracking relies on. Forget it and PyTorch raises an error as soon as you
# assign the first layer or parameter.
#
# `forward()` holds the computation. You never call it directly. You write
# `layer(x)`, which calls nn.Module's `__call__` method (the "dunder" hook Python
# runs when an object is used like a function), and `__call__` calls `forward(x)`
# for you after doing some bookkeeping (e.g. running "hooks"). So `layer(x)` is
# the right way, and `layer.forward(x)` skips the bookkeeping.
# -------------------------------------------------------------------------------------


class ClassTokenPosEmbed(nn.Module):
    """Insert a learnable class token at `cls_index`, then add learnable E_pos."""

    def __init__(self, num_patches=64, embed_dim=192, cls_index=32):
        # Parent constructor first, so nn.Module can register our parameters
        super().__init__()

        # The class token may go anywhere from the front (0) to the very end (J).
        # The default 32 is the middle of 64 patches (vim/config.py computes it).
        assert 0 <= cls_index <= num_patches, "cls_index must be in 0..num_patches"

        # Remember the settings for forward() and for anyone inspecting the module
        self.num_patches = num_patches
        self.cls_index = cls_index

        # -----------------------------------------------------------------------------
        # PYTORCH CONCEPT: nn.Parameter vs a plain tensor.
        # A plain tensor (torch.zeros(...)) stored on a module is just data: the
        # optimiser never sees it and it never changes during training. Wrapping it
        # in nn.Parameter(...) marks it as a LEARNABLE weight. Assigning a Parameter
        # to `self.something` inside an nn.Module registers it automatically, so it
        # appears in model.parameters() (and therefore gets updated by the
        # optimiser) and in model.state_dict() (and therefore gets saved).
        # Layers like nn.Linear create their Parameters internally; here we make
        # our own because the class token and E_pos are raw learnable tables, not
        # a layer that transforms an input.
        #
        # Why the leading dimension is 1: these are shared by every image in a
        # batch. We store ONE copy with a batch dimension of size 1, and let
        # broadcasting (see forward) stretch it to the batch size B for free.
        # -----------------------------------------------------------------------------

        # t_cls: one learnable vector of D = 192 numbers. Shape (1, 1, 192) means
        # (batch of 1, sequence of 1 token, 192 numbers).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # E_pos: one learnable D-vector per position. 64 patches + 1 class token = 65
        # positions, so shape (1, 65, 192).
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Fill both with small random numbers (see _init_weights for why).
        self._init_weights()

    def _init_weights(self):
        # A leading underscore in a name means "internal, not part of the public
        # interface". Python does not enforce it (there is no `private`); it is a
        # convention, like a comment saying "please do not call this from outside".

        # Truncated normal, std 0.02: draw from a bell curve with standard deviation
        # 0.02, re-drawing any value beyond 2 standard deviations, so every number
        # lies in [-0.04, 0.04].
        #
        # Why small RANDOM values and not zeros?
        #   - E_pos: if every position started with the same (zero) vector, every
        #     position would receive exactly the same gradient at first, and it
        #     would take a while for positions to become distinguishable. Random
        #     values make each position different from step one.
        #   - Both: SMALL values (0.02 versus patch tokens of size around 1) mean
        #     these additions do not drown out the image content at the start.
        # The trailing underscore in `trunc_normal_` is PyTorch's convention for
        # "modifies the tensor IN PLACE" rather than returning a new one.
        # Careful: `a` and `b` are ABSOLUTE cut-off values, not multiples of std.
        # Their defaults are -2 and 2, which at std 0.02 would cut off nothing
        # (that is 100 standard deviations), so we pass 2 * std explicitly.
        std = 0.02
        nn.init.trunc_normal_(self.cls_token, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.pos_embed, std=std, a=-2 * std, b=2 * std)

    def forward(self, patch_tokens):
        # patch_tokens: (B, 64, 192), the output of PatchEmbed
        # Check we received the number of patches we were built for
        assert patch_tokens.shape[1] == self.num_patches, (
            f"Expected {self.num_patches} patch tokens, got {patch_tokens.shape[1]}"
        )

        # B = how many images are in this batch (it can change from call to call,
        # e.g. the last batch of an epoch is often smaller)
        batch_size = patch_tokens.shape[0]

        # Make one class token per image: (1, 1, 192) -> (B, 1, 192).
        # `expand` does not copy memory; it creates a VIEW that pretends the size-1
        # dimension is B long (every image sees the same underlying 192 numbers).
        # -1 means "keep this dimension as it is". Gradients from all B copies flow
        # back and are summed into the single shared cls_token, as they should be.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Split the 64 patches into the 32 before the class token and the 32 after.
        # SLICING on dimension 1 (the sequence): `a[:, :32]` keeps every image (the
        # first `:`), and sequence positions 0..31 (`:32` stops BEFORE 32).
        # `a[:, 32:]` keeps positions 32..63.
        # (B, 64, 192) -> before (B, 32, 192) and after (B, 32, 192)
        before = patch_tokens[:, : self.cls_index]
        after = patch_tokens[:, self.cls_index :]

        # Glue the three pieces together along the sequence dimension (dim=1):
        # (B, 32, 192) + (B, 1, 192) + (B, 32, 192) -> (B, 65, 192).
        # The class token now sits at index 32, with 32 patches on each side.
        tokens = torch.cat([before, cls_tokens, after], dim=1)

        # Add the position embeddings: (B, 65, 192) + (1, 65, 192) -> (B, 65, 192).
        # PYTORCH CONCEPT: broadcasting. When two tensors are combined elementwise
        # and one of them has size 1 in some dimension, PyTorch acts as if that
        # dimension were repeated to match the other tensor (without copying). So
        # the same 65 x 192 table of positions is added to every image in the batch.
        tokens = tokens + self.pos_embed

        # Return T_0 of Eq. 4, ready for the first Vim block
        return tokens
