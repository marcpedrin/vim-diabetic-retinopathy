"""PatchEmbed: turns an image into a sequence of patch tokens.

Vim paper correspondence (arXiv 2401.09417, Section 3.2, Eq. 4):
    The paper takes a 2D image t of shape (H, W, C) and flattens it into
    J small 2D patches x_p (each P x P x C pixels). Each patch is then
    linearly projected to a D-dimensional vector by a learnable matrix W:

        T_0 = [ t_cls ; x_1 W ; x_2 W ; ... ; x_J W ] + E_pos

    This file implements ONLY the "x_p W" part: the linear projection W of
    the patches. (The class token t_cls and position embedding E_pos come
    in later steps.)

    Cutting into patches + linear projection is done in one go with a
    Conv2d whose kernel_size == stride == patch_size. Each kernel position
    sees exactly one patch and multiplies it by the same weights W.

Shapes (defaults): (B, 3, 128, 128) -> (B, 64, 192)
"""

import torch  # main PyTorch library (tensors)
import torch.nn as nn  # neural network building blocks (Module, Conv2d, ...)


class PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches and project each to D numbers."""

    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=192):
        # Call the parent class constructor so PyTorch can register our layers
        super().__init__()

        # The image must be cut evenly into patches, otherwise pixels are left over
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        # Remember the settings (useful for checks in forward())
        self.img_size = img_size
        self.patch_size = patch_size

        # Patches per side: 128 // 16 = 8
        grid = img_size // patch_size
        # Total patches J = 8 * 8 = 64
        self.num_patches = grid * grid

        # Conv2d acts as "cut into patches + multiply by W":
        #   in_chans  = 3 input channels (RGB)
        #   embed_dim = 192 output channels = D numbers per patch
        #   kernel_size = patch_size -> the window covers exactly one patch
        #   stride = patch_size      -> windows do not overlap
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x has shape (B, 3, H, W): batch, channels, height, width
        # Check that the input image is the size we were built for
        assert x.shape[2] == self.img_size and x.shape[3] == self.img_size, (
            f"Expected image size {self.img_size}x{self.img_size}, "
            f"got {x.shape[2]}x{x.shape[3]}"
        )

        # Apply the conv: (B, 3, 128, 128) -> (B, 192, 8, 8)
        x = self.proj(x)

        # Merge the 8x8 grid into one line of 64 patches: (B, 192, 8, 8) -> (B, 192, 64)
        x = x.flatten(2)

        # Swap the last two axes so each patch is a row: (B, 192, 64) -> (B, 64, 192)
        x = x.transpose(1, 2)

        # Return the patch tokens
        return x
