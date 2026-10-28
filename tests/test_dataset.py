"""Test the data pipeline on synthetic JPEGs written to a temporary folder.

What is checked:
    - the EyePACS dataset has the right length; one item is a (3, 128, 128)
      tensor with an integer label in 0..4
    - a missing image file is skipped with a warning instead of crashing
    - the black-border crop really removes the black frame
    - the train / val / test splits are disjoint, cover everything, and no
      patient appears in two splits (no leakage)
    - the split is reproducible (same seed -> same split)
    - a DataLoader batch has shape (B, 3, 128, 128)
    - the Messidor-2 dataset skips ungradable rows

PYTEST CONCEPT: fixtures. A test function that has a parameter named `tmp_path`
gets a brand-new, empty temporary folder from pytest (a pathlib.Path object),
which pytest cleans up later. pytest looks at the parameter NAME to decide what
to pass in; this is called a "fixture".
"""

import os  # for building file paths
import sys  # for editing the module search path

# Add the project root to Python's search path so `import data` works
# even when this file is run directly (python tests/test_dataset.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # test framework (pytest.warns)
import torch  # tensors
from PIL import Image  # for the crop test image

from data import build_dataloaders, build_messidor_loader  # under test
from data.eyepacs import EyePACSDataset, read_eyepacs_labels  # under test
from data.splits import split_by_patient  # under test
from data.synthetic import make_synthetic_eyepacs, make_synthetic_messidor
from data.transforms import CropToRetina, build_eval_transform  # under test
from vim.config import Config  # a default config object to point at the fake data


def make_config(image_dir, labels_csv):
    """A default Config whose EyePACS paths point at the synthetic data."""
    cfg = Config()
    cfg.data.eyepacs_image_dir = image_dir
    cfg.data.eyepacs_labels_csv = labels_csv
    # Load in the main process: simplest and fastest for a handful of images
    cfg.data.num_workers = 0
    return cfg


def test_dataset_length_and_item(tmp_path):
    # 10 patients x 2 eyes = 20 fake images in the temporary folder
    image_dir, labels_csv = make_synthetic_eyepacs(str(tmp_path), num_patients=10)
    labels = read_eyepacs_labels(labels_csv)

    dataset = EyePACSDataset(image_dir, labels, build_eval_transform(img_size=128))

    # len(dataset) calls __len__
    assert len(dataset) == 20

    # dataset[0] calls __getitem__(0)
    image, label = dataset[0]
    print("Item image shape:", tuple(image.shape), " label:", label)
    assert image.shape == (3, 128, 128)
    # A plain Python int, and a valid DR grade
    assert isinstance(label, int)
    assert 0 <= label <= 4
    # The patient id column was derived from names like "1_left"
    assert labels["patient_id"].iloc[0] == "1"


def test_missing_image_is_skipped_with_warning(tmp_path):
    image_dir, labels_csv = make_synthetic_eyepacs(str(tmp_path), num_patients=5)
    labels = read_eyepacs_labels(labels_csv)

    # Delete one image that the CSV still lists
    os.remove(os.path.join(image_dir, "3_left.jpeg"))

    # `with pytest.warns(UserWarning):` passes only if a warning is issued inside
    with pytest.warns(UserWarning):
        dataset = EyePACSDataset(image_dir, labels, build_eval_transform())

    # 10 listed, 1 missing -> 9 usable, and every remaining item still loads
    assert len(dataset) == 9
    for i in range(len(dataset)):
        image, _ = dataset[i]
        assert image.shape == (3, 128, 128)


def test_crop_removes_black_border():
    # A 300 x 200 black image with a white 100 x 80 rectangle at (120, 60)
    image = Image.new("RGB", (300, 200), (0, 0, 0))
    image.paste((255, 255, 255), (120, 60, 220, 140))

    cropped = CropToRetina(threshold=10)(image)

    # Exactly the bright rectangle is left
    assert cropped.size == (100, 80)


def test_splits_are_disjoint_and_patient_grouped(tmp_path):
    # 50 patients -> 100 images; enough for 10% val and test per grade
    _, labels_csv = make_synthetic_eyepacs(str(tmp_path), num_patients=50)
    labels = read_eyepacs_labels(labels_csv)

    train_df, val_df, test_df = split_by_patient(labels, 0.1, 0.1, seed=42)
    print("Split sizes (images):", len(train_df), len(val_df), len(test_df))

    # The image names in each split, as Python SETS (unordered, no duplicates,
    # like Java's HashSet), so we can intersect them
    train_imgs = set(train_df["image"])
    val_imgs = set(val_df["image"])
    test_imgs = set(test_df["image"])

    # The patient ids in each split
    train_pat = set(train_df["patient_id"])
    val_pat = set(val_df["patient_id"])
    test_pat = set(test_df["patient_id"])

    # No image in two splits (`&` on sets is the intersection)
    assert not (train_imgs & val_imgs)
    assert not (train_imgs & test_imgs)
    assert not (val_imgs & test_imgs)

    # No PATIENT in two splits: both eyes always travel together
    assert not (train_pat & val_pat)
    assert not (train_pat & test_pat)
    assert not (val_pat & test_pat)

    # Together they cover every image exactly once (`|` is the union)
    assert len(train_df) + len(val_df) + len(test_df) == len(labels)
    assert train_imgs | val_imgs | test_imgs == set(labels["image"])

    # Val and test are not empty, and train is the largest part
    assert len(val_df) > 0 and len(test_df) > 0
    assert len(train_df) > len(val_df)


def test_split_is_reproducible(tmp_path):
    _, labels_csv = make_synthetic_eyepacs(str(tmp_path), num_patients=50)
    labels = read_eyepacs_labels(labels_csv)

    # Same seed twice -> identical splits; a different seed -> a different one
    first = split_by_patient(labels, 0.1, 0.1, seed=42)
    second = split_by_patient(labels, 0.1, 0.1, seed=42)
    other = split_by_patient(labels, 0.1, 0.1, seed=7)

    assert list(first[1]["image"]) == list(second[1]["image"])
    assert list(first[1]["image"]) != list(other[1]["image"])


def test_dataloader_batch_shape(tmp_path):
    image_dir, labels_csv = make_synthetic_eyepacs(str(tmp_path), num_patients=20)
    cfg = make_config(image_dir, labels_csv)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, batch_size=4)

    # next(iter(loader)) fetches the first batch
    images, labels = next(iter(train_loader))
    print("Batch shapes:", tuple(images.shape), tuple(labels.shape))

    assert images.shape == (4, 3, 128, 128)
    assert labels.shape == (4,)
    # The DataLoader turned the Python int labels into a tensor of integers
    assert labels.dtype == torch.int64


def test_messidor_skips_ungradable(tmp_path):
    # 6 gradable images plus 1 ungradable row
    image_dir, labels_csv = make_synthetic_messidor(str(tmp_path), num_images=6)
    cfg = Config()
    cfg.data.messidor_image_dir = image_dir
    cfg.data.messidor_labels_csv = labels_csv
    cfg.data.num_workers = 0

    loader = build_messidor_loader(cfg, batch_size=3)

    # The ungradable row was dropped
    assert len(loader.dataset) == 6
    images, labels = next(iter(loader))
    assert images.shape == (3, 3, 128, 128)


# Allow running as a plain script: python tests/test_dataset.py
if __name__ == "__main__":
    # tempfile.TemporaryDirectory stands in for pytest's tmp_path here
    import tempfile

    test_crop_removes_black_border()
    # The tests that need a temporary folder, in a list. Functions are ordinary
    # values in Python, so they can be stored in a list and called later.
    folder_tests = [
        test_dataset_length_and_item,
        test_missing_image_is_skipped_with_warning,
        test_splits_are_disjoint_and_patient_grouped,
        test_split_is_reproducible,
        test_dataloader_batch_shape,
        test_messidor_skips_ungradable,
    ]
    for test in folder_tests:
        with tempfile.TemporaryDirectory() as folder:
            test(folder)
    print("OK")
