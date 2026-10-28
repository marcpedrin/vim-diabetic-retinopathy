"""VisionMamba: the complete Vim classifier, from image pixels to 5 grade scores.

What this file does:
    Chains every piece built so far into one model:
        image -> patches -> tokens -> 12 Vim blocks -> norm -> class token -> logits

Vim paper correspondence (arXiv 2401.09417, Section 3.2 Eq. 4, and Section 3.4):
    Section 3.2 describes the whole forward pass in five steps:
      1. Split the image into patches and project each:   t_p^j W
                                                          -> PatchEmbed
      2. Insert the class token and add positions:        T_0 = [...] + E_pos  (Eq. 4)
                                                          -> ClassTokenPosEmbed
      3. Run L Vim blocks:      T_l = Vim(T_{l-1}) + T_{l-1}   for l = 1 .. L
                                                          -> VimBlock x 12
         (the "+ T_{l-1}" residual is inside VimBlock.forward)
      4. Normalise the class token's output:   f = Norm(T_L^0)
                                                          -> self.norm, then index 32
         (T_L^0 is the paper's notation for "the class token of the last layer")
      5. Classify:                             p_hat = MLP(f)
                                                          -> self.head, a Linear D -> 5
    Section 3.4 (architecture details) gives Vim-Ti's sizes: D = 192, E = 384,
    N = 16, L = 24. We use L = 12 (see configs/default.yaml).

Shapes (defaults): images (B, 3, 128, 128) -> logits (B, 5)
    (B, 3, 128, 128) -> (B, 64, 192) -> (B, 65, 192) -> 12 x (B, 65, 192)
    -> (B, 65, 192) -> (B, 192) -> (B, 5)

Deliberate differences from the paper:
    1. 12 blocks instead of 24, and 128-pixel images instead of 224 (approved
       scale-down; see configs/default.yaml).
    2. The paper writes "MLP" for the head; the official code uses a single
       Linear layer, and so do we.
    3. No drop-path (stochastic depth) regularisation, which the official
       training recipe uses; it is a training trick, not part of the model.

Why classify from the class token rather than pooling all tokens (Table 5):
    The paper compared five ways of turning 65 output vectors into one:
        mean pool (average all tokens)       73.9% top-1
        max pool  (elementwise maximum)      73.4%
        head class token (index 0)           75.2%
        double class token (front and back)  74.3%
        middle class token (index 32)        76.1%   <- the default, used here
    Pooling treats every patch as equally important, but most of a fundus image
    is healthy background; a class token can LEARN, through the B/C/Delta gates,
    which patches to listen to. And the middle position is reached by both scan
    directions with the shortest average distance (see vim/tokens.py).
"""

import torch.nn as nn  # neural network building blocks (Module, ModuleList, ...)

# The four building blocks, from the other files of this package
from .patch_embed import PatchEmbed
from .tokens import ClassTokenPosEmbed
from .vim_block import VimBlock


class VisionMamba(nn.Module):
    """Vision Mamba image classifier: (B, 3, 128, 128) -> (B, 5) logits."""

    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=192,
                 expand=2, d_state=16, depth=12, num_classes=5, cls_index=32,
                 conv_kernel=4, dt_min=0.001, dt_max=0.1):
        # Parent constructor first, so nn.Module can register our layers
        super().__init__()

        # Step 1: image -> patch tokens, (B, 3, 128, 128) -> (B, 64, 192)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Remember where the class token sits; forward() reads it back out at the end
        self.cls_index = cls_index

        # Step 2: insert t_cls at index 32 and add E_pos (Eq. 4),
        # (B, 64, 192) -> (B, 65, 192)
        self.tokens = ClassTokenPosEmbed(
            num_patches=self.patch_embed.num_patches,
            embed_dim=embed_dim,
            cls_index=cls_index,
        )

        # Step 3: L = 12 Vim blocks, each (B, 65, 192) -> (B, 65, 192).
        # PYTORCH CONCEPT: nn.ModuleList is a Python list that nn.Module can see
        # into. A plain list `[VimBlock(), ...]` would run fine, but its blocks'
        # weights would be INVISIBLE to .parameters(), so the optimiser would never
        # train them and state_dict() would never save them. ModuleList registers
        # every element. Each VimBlock() call creates a NEW block with its own
        # weights: the 12 blocks share nothing.
        blocks = []
        for _ in range(depth):
            blocks.append(
                VimBlock(
                    embed_dim=embed_dim,
                    expand=expand,
                    d_state=d_state,
                    conv_kernel=conv_kernel,
                    dt_min=dt_min,
                    dt_max=dt_max,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Step 4: f = Norm(T_L^0). The blocks' residual additions mean the token
        # values can drift in scale through the stack; a final LayerNorm brings
        # them back to a standard scale before the classifier sees them.
        self.norm = nn.LayerNorm(embed_dim)

        # Step 5: p_hat = MLP(f). One Linear layer, D = 192 numbers -> 5 scores,
        # one per DR grade. The scores ("logits") are raw real numbers; softmax,
        # applied inside the loss function, turns them into probabilities.
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, images):
        # images: (B, 3, 128, 128), normalised RGB fundus photographs

        # Step 1: (B, 3, 128, 128) -> (B, 64, 192)
        x = self.patch_embed(images)

        # Step 2 (Eq. 4): (B, 64, 192) -> (B, 65, 192), class token at index 32
        x = self.tokens(x)

        # Step 3: 12 blocks in order, each (B, 65, 192) -> (B, 65, 192).
        # Iterating over a ModuleList gives the blocks one by one.
        for block in self.blocks:
            x = block(x)

        # Step 4a: normalise every token: (B, 65, 192) -> (B, 65, 192)
        x = self.norm(x)

        # Step 4b: keep ONLY the class token. Indexing dimension 1 with a single
        # number removes that dimension: (B, 65, 192) -> (B, 192).
        # The other 64 tokens have done their job (passing information into the
        # class token through the scans) and are discarded.
        cls_output = x[:, self.cls_index]

        # Step 5: (B, 192) -> (B, 5), one score per DR grade
        logits = self.head(cls_output)

        # Raw scores; the loss (training) or argmax (prediction) comes later
        return logits


def build_model(model_cfg):
    """Build a VisionMamba from the `model` section of the config (vim/config.py).

    Keeping this in one function means training, evaluation and tests all build
    the model in exactly the same way from the same settings.
    """
    # Every argument comes from configs/default.yaml; nothing is hard-coded here.
    # cls_index and the expand factor are read from the config object; cls_index
    # is a derived value (@property) computed from cls_position.
    return VisionMamba(
        img_size=model_cfg.img_size,
        patch_size=model_cfg.patch_size,
        in_chans=model_cfg.in_chans,
        embed_dim=model_cfg.embed_dim,
        expand=model_cfg.expand,
        d_state=model_cfg.d_state,
        depth=model_cfg.depth,
        num_classes=model_cfg.num_classes,
        cls_index=model_cfg.cls_index,
        conv_kernel=model_cfg.conv_kernel,
        dt_min=model_cfg.dt_min,
        dt_max=model_cfg.dt_max,
    )


def count_parameters(module):
    """Count the learnable numbers (weights) in a module.

    Our total with the default config is 6,857,093 (about 6.9M), compared with
    the paper's 7M for Vim-Ti. You might expect about half the paper's size,
    since we use half the depth (12 blocks instead of 24). It is NOT half,
    because of one layer: the Delta projection. The printed Algorithm 1
    (line 11) maps E -> E directly, which costs 384 x 384 = 147,456 weights per
    direction per block. The official code factorises it through a rank of 12
    (E -> 12 -> E), so its whole B/C/Delta projection costs about 22,000
    weights per direction instead of our 160,128. Per block:
        ours (printed algorithm)   557,952 parameters
        official (low-rank Delta)  about 282,000 parameters
    So our 12 blocks hold about as many weights as 24 official blocks, and the
    two Delta projections are 53% of each of our blocks. Our model is therefore
    NOT smaller than Vim-Ti in parameters, only in depth and in compute per
    image (fewer blocks, and 65 tokens instead of 197 at 224 pixels). That is
    worth a sentence in the report as a consequence of following the printed
    algorithm. (The smaller image size barely changes the count: only E_pos
    depends on the number of patches.)

    Breakdown (run `python -m vim.vision_mamba` to print it):
        patch_embed 147,648 | tokens 12,672 | 12 blocks 6,695,424 |
        final norm 384 | head 965
    """
    total = 0
    # .parameters() yields every learnable tensor: weights, biases, A_log, E_pos...
    for p in module.parameters():
        # requires_grad is True for anything the optimiser is allowed to change
        if p.requires_grad:
            # numel() = "number of elements", e.g. 384 * 16 = 6144 for an A_log
            total += p.numel()
    return total


# PYTHON CONCEPT: this block runs only when the file is executed directly
# (`python -m vim.vision_mamba`), not when it is imported by another file.
# __name__ is "__main__" for the file Python was asked to run.
if __name__ == "__main__":
    # Build the default model and print where its parameters live
    model = VisionMamba()
    print(f"Total parameters: {count_parameters(model):,}")
    # named_children() gives the top-level parts: patch_embed, tokens, blocks, ...
    for name, child in model.named_children():
        print(f"  {name:12s} {count_parameters(child):>10,}")
    # The cost of one block, split by part
    block = model.blocks[0]
    print("One VimBlock:")
    for name, child in block.named_children():
        print(f"  {name:12s} {count_parameters(child):>10,}")
