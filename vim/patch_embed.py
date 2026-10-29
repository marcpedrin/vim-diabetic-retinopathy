"""
PatchEmbed: turns an image into a sequence of patch tokens.

    The paper takes a 2D image t of shape (H, W, C) and flattens it into
    J small 2D patches x_p (each P x P x C pixels). 

"""

import torch
import torch.nn as nn  


class PatchEmbed(nn.Module):
    #Split an image into non-overlapping patches and project each to D numbers.

    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        grid = img_size // patch_size
        #Total patches J = 8 * 8 = 64
        self.num_patches = grid * grid

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            #RGB
            out_channels=embed_dim,
            #192 output
            kernel_size=patch_size,
            stride=patch_size,
            #shifting
        )

    def forward(self, x):
        #x has shape (B, 3, H, W): batch, channels, height, width
        assert x.shape[2] == self.img_size and x.shape[3] == self.img_size, (
            f"Expected image size {self.img_size}x{self.img_size}, "
            f"got {x.shape[2]}x{x.shape[3]}"
        )

        #(B, 3, 128, 128) -> (B, 192, 8, 8) from linr 26
        x = self.proj(x)
        #8x8 grid into one line of 64 patches: (B, 192, 8, 8) to (B, 192, 64)
        x = x.flatten(2)
        #Swap the last two axes so each patch is a row: (B, 192, 64) -> (B, 64, 192)
        x = x.transpose(1, 2)
        return x
