"""Data package: datasets, preprocessing, splitting, and the DataLoader builders.

What this file does:
    build_dataloaders(cfg)      -> (train_loader, val_loader, test_loader) for EyePACS
    build_messidor_loader(cfg)  -> one loader for Messidor-2 (evaluation only)

Vim paper correspondence (arXiv 2401.09417):
    None; data handling only.

Shapes:
    every batch is (images, labels) with images (B, 3, 128, 128) and labels (B,)

Deliberate differences from the paper:
    Not applicable.

What a DataLoader does (and the four settings we choose):
    A Dataset answers "give me item i". A DataLoader sits on top and, each
    epoch, decides the ORDER of the items, asks the Dataset for them, and glues
    every `batch_size` of them into one batch: B tensors of (3, 128, 128) are
    stacked into one (B, 3, 128, 128) tensor, and B ints into a (B,) tensor.
      batch_size   images per optimisation step (32 by default). Larger batches
                   give smoother gradients but need more GPU memory.
      shuffle      True for training: a NEW random order every epoch, so the model
                   never sees the images in a fixed sequence it could exploit, and
                   each batch is a fair mix of grades. False for val/test: order
                   does not matter there, and a fixed order is reproducible.
      num_workers  how many background processes load and preprocess images
                   while the GPU trains on the previous batch. 0 = load in the
                   main process (simplest; used by the tests).
      pin_memory   copies batches into "page-locked" RAM, which the GPU can read
                   faster. Only useful with a GPU, so we switch it off on CPU.
"""

import torch  # torch.Generator, torch.cuda.is_available
from torch.utils.data import DataLoader  # batches and shuffles a Dataset

# Seeds NumPy/Python inside each loader worker (see train/utils.py)
from train.utils import seed_worker

# The pieces defined in this package
from .eyepacs import EyePACSDataset, read_eyepacs_labels
from .messidor2 import Messidor2Dataset
from .splits import split_by_patient
from .transforms import build_eval_transform, build_train_transform


def _make_loader(dataset, batch_size, shuffle, data_cfg, seed):
    """Wrap a Dataset in a DataLoader with our standard, reproducible settings."""
    # pin_memory only makes sense if there is a GPU to copy to
    pin_memory = data_cfg.pin_memory and torch.cuda.is_available()

    # The generator controls the shuffle order; seeding it makes the order of every
    # epoch reproducible (epoch 1 always sees the same order, epoch 2 another, ...)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_cfg.num_workers,
        pin_memory=pin_memory,
        # Runs once in each worker process when it starts (see train/utils.py)
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_dataloaders(cfg, max_images=None, batch_size=None):
    """Build the EyePACS train / val / test loaders from the config.

    Args:
        cfg: the full Config (vim/config.py).
        max_images: if given, use only the first this-many rows of the CSV
                    (the --smoke-test uses 100).
        batch_size: overrides cfg.train.batch_size if given.
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Read trainLabels.csv: columns image, level, patient_id
    labels = read_eyepacs_labels(cfg.data.eyepacs_labels_csv)

    # For a quick run, keep only the first rows. The CSV is ordered by patient,
    # so both eyes of a patient usually stay together; the split below groups by
    # patient anyway, so even a lone eye cannot leak.
    if max_images is not None:
        labels = labels.head(max_images)

    # Patient-level, stratified, seeded split (data/splits.py)
    train_df, val_df, test_df = split_by_patient(
        labels,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        seed=cfg.seed,
    )

    # Training images get random augmentation; val/test get the fixed pipeline
    train_tf = build_train_transform(
        img_size=cfg.model.img_size,
        crop_threshold=cfg.data.crop_threshold,
        rotation_degrees=cfg.data.rotation_degrees,
        color_jitter=cfg.data.color_jitter,
    )
    eval_tf = build_eval_transform(
        img_size=cfg.model.img_size,
        crop_threshold=cfg.data.crop_threshold,
    )

    # One Dataset per part, all reading from the same image folder
    image_dir = cfg.data.eyepacs_image_dir
    ext = cfg.data.eyepacs_image_ext
    train_set = EyePACSDataset(image_dir, train_df, train_tf, image_ext=ext)
    val_set = EyePACSDataset(image_dir, val_df, eval_tf, image_ext=ext)
    test_set = EyePACSDataset(image_dir, test_df, eval_tf, image_ext=ext)

    # `x if condition else y` is Python's ternary operator (Java: cond ? x : y)
    bs = batch_size if batch_size is not None else cfg.train.batch_size

    # Shuffle only the training set
    train_loader = _make_loader(train_set, bs, True, cfg.data, cfg.seed)
    val_loader = _make_loader(val_set, bs, False, cfg.data, cfg.seed)
    test_loader = _make_loader(test_set, bs, False, cfg.data, cfg.seed)

    return train_loader, val_loader, test_loader


def build_messidor_loader(cfg, batch_size=None):
    """Build the Messidor-2 loader. EVALUATION ONLY (see data/messidor2.py)."""
    # Always the deterministic eval pipeline: no augmentation on a test set
    eval_tf = build_eval_transform(
        img_size=cfg.model.img_size,
        crop_threshold=cfg.data.crop_threshold,
    )
    dataset = Messidor2Dataset(
        cfg.data.messidor_image_dir,
        cfg.data.messidor_labels_csv,
        eval_tf,
        id_column=cfg.data.messidor_id_column,
        label_column=cfg.data.messidor_label_column,
        default_ext=cfg.data.messidor_image_ext,
    )
    bs = batch_size if batch_size is not None else cfg.train.batch_size
    # Never shuffled: it is only ever evaluated
    return _make_loader(dataset, bs, False, cfg.data, cfg.seed)
