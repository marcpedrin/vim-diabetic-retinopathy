"""Messidor-2 dataset: an EXTERNAL test set, for evaluation only.

#####################################################################################
#                                                                                   #
#   WARNING: Messidor-2 must NEVER be used for training or fine-tuning.             #
#                                                                                   #
#   This is an explicit rule of the task. Messidor-2 exists in this project to      #
#   measure how well a model trained on EyePACS GENERALISES to images from          #
#   different cameras, clinics and patients. If any Messidor-2 image influenced     #
#   the weights (training, fine-tuning, choosing a checkpoint, tuning a threshold   #
#   or a hyperparameter), the cross-dataset result would be meaningless.            #
#   Only eval/evaluate.py may use this file.                                        #
#                                                                                   #
#####################################################################################

What this file does:
    Messidor2Dataset returns (image_tensor, grade) pairs for the Messidor-2
    images, using the same three-method Dataset contract as data/eyepacs.py
    (see the explanation there).

Vim paper correspondence (arXiv 2401.09417):
    None. Messidor-2 is our external evaluation set.

Shapes:
    one item: image tensor (3, 128, 128), label a Python int in 0..4

Deliberate differences from the paper:
    Not applicable (different dataset).

About the data:
    1,748 fundus images (874 examinations, both eyes) from French hospitals,
    a different camera and population from EyePACS. The DR grades usually used
    with it are the adjudicated 0-4 grades released by Krause et al. (2018),
    distributed on Kaggle as a CSV whose columns include
        image_id, adjudicated_dr_grade, adjudicated_gradable
    A few images were judged ungradable and have no grade; they are skipped.
    The column names are config settings (data.messidor_id_column and
    data.messidor_label_column), so a copy with different names needs only a
    config change.
"""

import os  # file paths and existence checks
import warnings  # print a warning without stopping the program

import pandas as pd  # reads the CSV of grades
from torch.utils.data import Dataset  # PyTorch's base class for datasets

# Reuse the same fast, safe image loader as EyePACS, so both datasets are opened
# in exactly the same way
from .eyepacs import load_rgb_image


def resolve_image_path(image_dir, image_id, default_ext):
    """Build the file path for one image id.

    In the common Messidor-2 CSV the ids already carry their extension
    ("20051020_43808_0100_PP.png"); and the image files mix .png, .jpg and .JPG.
    So: keep the id's own extension if it has one, otherwise add `default_ext`.
    """
    # os.path.splitext("a.png") -> ("a", ".png"); for "a" it gives ("a", "")
    _, ext = os.path.splitext(image_id)
    # An empty string counts as False in an `if`, so this means "has no extension"
    if not ext:
        image_id = image_id + default_ext
    return os.path.join(image_dir, image_id)


class Messidor2Dataset(Dataset):
    """Messidor-2 images and adjudicated DR grades. EVALUATION ONLY."""

    def __init__(self, image_dir, labels_csv, transform=None,
                 id_column="image_id", label_column="adjudicated_dr_grade",
                 default_ext=".png", draft_size=512):
        # Parent constructor
        super().__init__()

        # How to process each image (always the deterministic eval transform)
        self.transform = transform
        self.draft_size = draft_size

        # Read the grades table
        table = pd.read_csv(labels_csv)

        # Fail early and clearly if the configured columns are not there
        for column in (id_column, label_column):
            if column not in table.columns:
                raise ValueError(
                    f"{labels_csv} has no '{column}' column; set "
                    f"data.messidor_id_column / data.messidor_label_column in the "
                    f"config to the names your copy uses"
                )

        # Ungradable images have an empty grade, which pandas reads as NaN ("not a
        # number"). notna() is True where a grade exists; keep only those rows.
        gradable = table[table[label_column].notna()]
        skipped_ungradable = len(table) - len(gradable)

        # Build (path, label) pairs, skipping missing files, as in EyePACSDataset
        self.samples = []
        missing = []
        for image_id, grade in zip(gradable[id_column], gradable[label_column]):
            path = resolve_image_path(image_dir, str(image_id), default_ext)
            if os.path.exists(path):
                self.samples.append((path, int(grade)))
            else:
                missing.append(path)

        # One message for each kind of skipped row
        if skipped_ungradable:
            print(f"Messidor2Dataset: skipped {skipped_ungradable} ungradable image(s)")
        if missing:
            warnings.warn(
                f"Messidor2Dataset: {len(missing)} image(s) listed in the CSV were "
                f"not found and will be skipped (e.g. {missing[0]})"
            )

        # The labels alone, in sample order
        self.labels = [label for _, label in self.samples]

    def __len__(self):
        # Number of usable (image, label) pairs
        return len(self.samples)

    def __getitem__(self, index):
        # Which file and grade this index refers to
        path, label = self.samples[index]

        # Read the image now; a corrupt file raises an error naming it
        try:
            image = load_rgb_image(path, self.draft_size)
        except OSError as error:
            raise RuntimeError(f"Could not read image {path}") from error

        # Same preprocessing as the EyePACS test set -> tensor (3, 128, 128)
        if self.transform is not None:
            image = self.transform(image)

        # One (image_tensor, label) pair
        return image, label
