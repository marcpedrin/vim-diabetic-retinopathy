"""EyePACS dataset: reads the Kaggle DR images and labels, one (image, grade) at a time.

What this file does:
    - read_eyepacs_labels() reads Kaggle's trainLabels.csv into a table with one
      row per image: file name, DR grade (0-4) and patient id.
    - EyePACSDataset is a torch Dataset over (some of) those rows: asking it for
      item i opens image i from disk, preprocesses it, and returns
      (image_tensor, grade).
    - load_rgb_image() opens an image file quickly and safely (shared with
      data/messidor2.py).

Vim paper correspondence (arXiv 2401.09417):
    None. The paper trains on ImageNet; this is our task's training data.

Shapes:
    one item: image tensor (3, 128, 128), label a Python int in 0..4

Deliberate differences from the paper:
    Not applicable (different dataset).

About the data (Kaggle "Diabetic Retinopathy Detection", 2015, from EyePACS):
    35,126 labelled training images from 17,563 patients, LEFT and RIGHT eye
    for each. trainLabels.csv has two columns:
        image,level
        10_left,0
        10_right,0
        13_left,0
    "10_left" is patient 10's left eye, stored as 10_left.jpeg. `level` is the
    DR grade on the international clinical scale:
        0 no DR, 1 mild, 2 moderate, 3 severe non-proliferative, 4 proliferative
    About 73% of the images are grade 0, which is why the loss uses class
    weights (train/train.py) and why plain accuracy is a misleading metric
    (eval/metrics.py).

No ready-made loader:
    The task forbids e.g. torchvision.datasets.ImageFolder, so this file does
    the work itself: read the CSV with pandas, build file paths, open images
    with Pillow.
"""

import os  # file paths and existence checks
import warnings  # standard library: print a warning without stopping the program

import pandas as pd  # pandas: reads CSV files into tables ("DataFrames")
from PIL import Image  # Pillow: opening image files

# torch.utils.data.Dataset is PyTorch's base class for datasets (explained below)
from torch.utils.data import Dataset


def read_eyepacs_labels(csv_path):
    """Read Kaggle's trainLabels.csv into a DataFrame with a patient_id column.

    Returns a pandas DataFrame (a table, like a spreadsheet) with columns:
        image       e.g. "10_left"    (file name without extension)
        level       e.g. 0            (DR grade, int 0..4)
        patient_id  e.g. "10"         (everything before the "_left"/"_right")
    """
    # Read the whole CSV; the header line gives the column names.
    # dtype={"image": str} keeps names like "10_left" as text (they already are,
    # but being explicit avoids surprises with purely numeric names).
    table = pd.read_csv(csv_path, dtype={"image": str})

    # Fail early and clearly if this is not the file we expect
    for column in ("image", "level"):
        if column not in table.columns:
            raise ValueError(f"{csv_path} has no '{column}' column")

    # The patient id is the part of the name before the last underscore:
    # "10_left" -> "10". `.str` gives string methods for a whole column at once;
    # rsplit("_", n=1) splits at the LAST "_" only, and .str[0] takes the left part.
    table["patient_id"] = table["image"].str.rsplit("_", n=1).str[0]

    # Make sure grades are plain integers (not floats such as 2.0)
    table["level"] = table["level"].astype(int)

    return table


def load_rgb_image(path, draft_size=None):
    """Open an image file and return it as an RGB PIL image.

    `draft_size` is a SPEED optimisation for JPEG files only: EyePACS photos can be
    5000 x 3000 pixels, and decoding all of them just to shrink them to 128 x
    128 is slow. Pillow's draft() asks the JPEG decoder to decode directly at a
    reduced scale (1/2, 1/4 or 1/8), never smaller than `draft_size`. It has no
    effect on other formats.
    """
    # `with` closes the file once we have made our copy (see vim/config.py)
    with Image.open(path) as image:
        # Ask for a cheap reduced-size decode, if wanted
        if draft_size is not None:
            image.draft("RGB", (draft_size, draft_size))
        # convert("RGB") forces 3 colour channels (some files are grey-scale or have
        # a 4th transparency channel); it also loads the pixels, so the result
        # stays valid after the file is closed.
        return image.convert("RGB")


# -------------------------------------------------------------------------------------
# PYTORCH CONCEPT: the Dataset contract (three methods)
#
# torch.utils.data.Dataset is a base class that promises "you can ask me how many
# items I have, and for item number i". A subclass must provide:
#   __init__(...)       set up: find out what the items are (here: a list of
#                       (file path, label) pairs). Do NOT load all the images here;
#                       35,000 full-size photos would not fit in memory.
#   __len__()           return the number of items. Python calls it for len(dataset).
#   __getitem__(i)      return item i, here one (image_tensor, label) pair. Python
#                       calls it for dataset[i]. This is where an image is actually
#                       read from disk, one at a time, only when it is needed.
# (These are "dunder" methods: Python calls them for you in response to len(...)
# and [...], just as Java calls hashCode() for you inside a HashMap.)
#
# The Dataset only knows about ONE item at a time. Grouping items into batches,
# shuffling their order every epoch, and loading them in parallel background
# processes is the job of the DataLoader (see data/__init__.py), which calls
# __getitem__ for you.
# -------------------------------------------------------------------------------------


class EyePACSDataset(Dataset):
    """EyePACS images and DR grades, as (image_tensor, label) pairs."""

    def __init__(self, image_dir, labels, transform=None, image_ext=".jpeg",
                 draft_size=512):
        """
        Args:
            image_dir: folder containing the .jpeg files.
            labels: a DataFrame from read_eyepacs_labels(), or a subset of it
                    (data/splits.py passes the train / val / test subsets).
            transform: a function applied to each PIL image (data/transforms.py).
            image_ext: extension added to each name in the CSV.
            draft_size: fast JPEG decode size; None decodes at full resolution.
        """
        # Parent constructor (Dataset's is trivial, but calling it is good practice)
        super().__init__()

        # Remember how to process each image
        self.transform = transform
        self.draft_size = draft_size

        # Build the list of samples, skipping any image that is missing on disk.
        # Handling this here, once, means a missing file produces one clear warning
        # instead of crashing the training run hours later in the middle of an epoch.
        self.samples = []
        missing = []
        # PYTHON CONCEPT: zip(a, b) walks two sequences side by side, giving pairs
        # (a[0], b[0]), (a[1], b[1]), ...; here: each image name with its grade.
        for name, level in zip(labels["image"], labels["level"]):
            # "10_left" -> "<image_dir>/10_left.jpeg"
            path = os.path.join(image_dir, name + image_ext)
            if os.path.exists(path):
                # A TUPLE (path, label): a small fixed group of values
                self.samples.append((path, int(level)))
            else:
                missing.append(path)

        # Report missing files once, with a count and an example, then carry on.
        if missing:
            warnings.warn(
                f"EyePACSDataset: {len(missing)} image(s) listed in the labels were "
                f"not found and will be skipped (e.g. {missing[0]})"
            )

        # The labels alone, in sample order. Training uses them to compute class
        # weights without having to open a single image.
        self.labels = [label for _, label in self.samples]

    def __len__(self):
        # Number of usable (image, label) pairs
        return len(self.samples)

    def __getitem__(self, index):
        # Look up which file and grade this index refers to (tuple unpacking)
        path, label = self.samples[index]

        # Read the image from disk now, only when it is asked for. A file that
        # exists but cannot be decoded (corrupt download) raises an error naming the
        # file, so the culprit is easy to find and remove.
        try:
            image = load_rgb_image(path, self.draft_size)
        except OSError as error:
            # `raise ... from error` keeps the original error attached for debugging
            raise RuntimeError(f"Could not read image {path}") from error

        # Crop, resize, augment and normalise -> tensor (3, 128, 128)
        if self.transform is not None:
            image = self.transform(image)

        # One (image_tensor, label) pair; the DataLoader stacks many into a batch
        return image, label
