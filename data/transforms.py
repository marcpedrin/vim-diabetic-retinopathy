"""Fundus image preprocessing and augmentation.

What this file does:
    Turns one retinal photograph (a PIL image of any size) into a normalised
    (3, 128, 128) tensor, the input format the model expects. Training images
    are also randomly flipped, rotated and colour-jittered (augmentation).

Vim paper correspondence (arXiv 2401.09417):
    None of the model maths. Section 4.1 lists the paper's ImageNet
    augmentations (random crop, horizontal flip, mixup, random erasing, label
    smoothing). We replace them with fundus-appropriate ones, see below.

Shapes:
    PIL image (H, W) of any size, e.g. 3888 x 2592 -> tensor (3, 128, 128)

Deliberate differences from the paper:
    - No random cropping and no random erasing. In ImageNet the object usually
      fills the picture and a crop still shows "a dog". In a fundus photo, the
      evidence for the grade can be a handful of microaneurysms, each a few
      pixels across. What separates grade 1 (mild) from grade 2 (moderate) is
      often how many of these tiny lesions there are and where; a random crop
      or an erased rectangle can remove exactly those, turning a correctly
      labelled grade-2 image into what LOOKS like a grade 1. That would teach
      the model a wrong label. Flips and small rotations never remove anything.
    - No mixup (blending two images and their labels). Half a grade-0 eye plus
      half a grade-4 eye is not a meaningful "grade 2" retina.
    - Mild colour jitter only, and no hue change: the colour of a lesion carries
      meaning (red haemorrhages versus yellow hard exudates).

The pipeline, in order (both for training and evaluation unless marked):
    1. CropToRetina   cut away the black frame around the circular retina
    2. PadToSquare    pad with black to a square, so resizing does not squash
    3. Resize         to 128 x 128
    4. (train only)   random horizontal flip, vertical flip, small rotation,
                      mild brightness/contrast/saturation jitter
    5. ToTensor       PIL image (0-255) -> float tensor (3, 128, 128) in [0, 1]
    6. Normalize      subtract ImageNet mean, divide by ImageNet std, per channel
"""

import numpy as np  # NumPy arrays, used to find the non-black pixels
from PIL import Image  # Pillow: opening, cropping and pasting images

# torchvision.transforms holds ready-made image operations (resize, flips, ...).
# These are allowed: the restriction is on ready-made DATASETS and MODELS.
from torchvision import transforms

# The per-channel (red, green, blue) mean and standard deviation of the ImageNet
# training set, in the 0-1 range. They are the de-facto standard normalisation for
# natural images: almost every vision codebase uses these exact numbers, so they
# make the preprocessing easy to reproduce. Normalising shifts each channel to
# roughly mean 0 and spread 1, which keeps the first layer's inputs on a scale where
# the default weight initialisation and the learning rate behave well.
# Honest caveat: fundus photos are redder and darker than ImageNet photos, so after
# this normalisation they are not exactly mean 0. Because we use no pretrained
# weights, statistics computed on EyePACS itself would be equally valid; what
# really matters is using the SAME numbers for training and testing.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropToRetina:
    """Crop away the black border around the circular retina.

    Fundus cameras photograph a bright circle (the retina, seen through the
    pupil) in the middle of a black rectangle. The black area carries no
    information but takes up a large, variable share of the image (it depends
    on the camera). Cropping to the retina means that after resizing to 128 x
    128, every one of our few pixels is spent on actual retina, and images from
    different cameras end up framed alike.

    PYTHON CONCEPT: `__call__`. Defining this method lets an OBJECT be called like
    a function: `crop = CropToRetina(10)` then `crop(image)`. torchvision's
    Compose (below) simply calls each step in turn, so any class with __call__
    can be one of its steps.
    """

    def __init__(self, threshold=10):
        # A pixel counts as "retina" if its grey level is above this (0-255 scale).
        # Slightly above 0 because JPEG compression leaves faint noise in the black.
        self.threshold = threshold

    def __call__(self, image):
        # Grey-scale copy, as a NumPy array of shape (H, W) with values 0-255.
        # "L" is Pillow's name for an 8-bit grey-scale mode (L for luminance).
        grey = np.asarray(image.convert("L"))

        # True where the pixel is bright enough to be retina: (H, W) of booleans
        mask = grey > self.threshold

        # Which rows / columns contain at least one retina pixel?
        # mask.any(axis=1): for each row, is ANY pixel in it True? -> shape (H,)
        # np.where(...)[0] gives the indices where that is True.
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]

        # A completely black (or failed) image: nothing to crop to, return it as is
        # rather than crash. `len(...) == 0` tests for an empty array.
        if len(rows) == 0 or len(cols) == 0:
            return image

        # Bounding box of the retina. Pillow's crop wants (left, top, right, bottom)
        # with right/bottom EXCLUSIVE, hence the + 1. rows[0] is the first bright
        # row and rows[-1] the last one; int() turns NumPy integers into Python ones.
        left = int(cols[0])
        top = int(rows[0])
        right = int(cols[-1]) + 1
        bottom = int(rows[-1]) + 1
        return image.crop((left, top, right, bottom))


class PadToSquare:
    """Pad an image with black to make it square, keeping it centred.

    Many fundus images are wider than tall even after cropping, because the
    camera clips the top and bottom of the retinal circle. Resizing a 1.3:1
    image straight to 128 x 128 would squash the round retina into an oval and
    distort lesion shapes. Padding first preserves the proportions; the added
    black matches the existing background.
    """

    def __call__(self, image):
        # Current size; Pillow reports (width, height)
        width, height = image.size
        # The square's side is the longer of the two
        side = max(width, height)
        # A black square canvas in the same colour mode (e.g. RGB)
        canvas = Image.new(image.mode, (side, side), 0)
        # Paste the image in the middle: offset = half the leftover space
        left = (side - width) // 2
        top = (side - height) // 2
        canvas.paste(image, (left, top))
        return canvas


def build_train_transform(img_size=128, crop_threshold=10, rotation_degrees=15.0,
                          color_jitter=0.1):
    """The training pipeline: preprocessing plus random augmentation.

    Augmentation shows the model a slightly different version of each image
    every epoch. With ~28,000 training images and 6.9M parameters, the model
    could otherwise start memorising individual photos instead of learning
    what lesions look like.
    """
    # transforms.Compose([...]) chains the steps: the output of one is the input of
    # the next. The square brackets make a Python LIST of the steps, in order.
    return transforms.Compose(
        [
            # 1-2. Remove the black frame, then make the image square
            CropToRetina(crop_threshold),
            PadToSquare(),
            # 3. Shrink to the model's input size. Doing this BEFORE the random steps
            #    makes them much cheaper (they then work on 128 x 128 pixels).
            transforms.Resize((img_size, img_size)),
            # 4a. Mirror left-right with probability 0.5. A left eye looks like a
            #     mirrored right eye, and DR grade does not depend on which eye.
            transforms.RandomHorizontalFlip(p=0.5),
            # 4b. Mirror top-bottom with probability 0.5. Lesions can appear
            #     anywhere on the retina, and the grade does not change if the
            #     picture is upside down.
            transforms.RandomVerticalFlip(p=0.5),
            # 4c. Rotate by a random angle in [-15, 15] degrees. Real photos are
            #     taken with the head slightly tilted; the corners revealed by the
            #     rotation are filled with black, like the existing background.
            transforms.RandomRotation(degrees=rotation_degrees),
            # 4d. Randomly vary brightness, contrast and saturation by up to 10%.
            #     Fundus photos differ in exposure between cameras and clinics; this
            #     teaches the model not to rely on overall brightness. Hue is left
            #     alone, because lesion COLOUR is diagnostic.
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            ),
            # 5. PIL image with values 0-255 -> float tensor (3, 128, 128) in [0, 1],
            #    channels first, as PyTorch layers expect
            transforms.ToTensor(),
            # 6. (value - mean) / std for each colour channel
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(img_size=128, crop_threshold=10):
    """The validation / test pipeline: the same preprocessing, NO randomness.

    Evaluation must be deterministic: the same image must always give the same
    tensor, so that a metric measured twice gives the same number.
    """
    return transforms.Compose(
        [
            # Steps 1-3 and 5-6 exactly as in training, so train and test images
            # look alike to the model; step 4 (augmentation) is left out
            CropToRetina(crop_threshold),
            PadToSquare(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
