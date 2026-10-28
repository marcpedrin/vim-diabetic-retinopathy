"""Training utilities: random seeding, running averages, checkpoints, CSV logging.

What this file does:
    Small helpers used by the training and evaluation scripts:
      - set_seed()        makes every random number generator deterministic
      - seed_worker()     does the same inside DataLoader background workers
      - AverageMeter      keeps a running average (e.g. mean loss over an epoch)
      - save_checkpoint() / load_checkpoint()   write / read model weights to disk
      - Logger            appends one row of metrics per epoch to a CSV file

Vim paper correspondence (arXiv 2401.09417):
    None of this is part of the Vim architecture. It is engineering that serves
    grading criterion (3), reproducibility: anyone re-running the code with the
    same config and seed should see the same numbers.

Input / output:
    No model tensors flow through this file, except the state dictionaries
    (name -> weight tensor) stored inside checkpoints.

Deliberate differences from the paper:
    None. (The paper does not describe its seeding or logging.)

Why determinism matters here:
    A neural network starts from RANDOM weights, sees the training images in a
    RANDOM order, and is shown RANDOMLY flipped/rotated copies of them. Run it
    twice without fixing the seeds and you get two different final scores. The
    examiner then cannot tell whether a change you made helped, or whether you
    just got luckier dice. Fixing every seed makes a run repeatable, which is
    exactly what the "reproducibility" criterion asks for.

    Honest caveat: bit-for-bit identical results are only guaranteed on the
    same hardware and library versions. Different GPU models can round floating
    point numbers in a slightly different order.
"""

import csv  # standard library: reading and writing comma-separated value files
import os  # standard library: file paths, folders and environment variables
import random  # standard library: Python's own random number generator

import numpy as np  # NumPy: fast arrays; `as np` gives the module a short nickname
import torch  # PyTorch: tensors, neural networks and automatic differentiation


def set_seed(seed: int) -> None:
    """Seed every random number generator the project touches.

    There are FOUR separate generators, and each has to be seeded on its own:
    Python's `random`, NumPy's, PyTorch's CPU generator and PyTorch's GPU ones.
    """
    # Python's hash() of strings is randomised per process unless this is fixed.
    # It affects e.g. the iteration order of sets of strings. Setting it here only
    # affects child processes (like DataLoader workers) started after this line.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 1) Python's built-in generator (used by e.g. random.shuffle).
    random.seed(seed)

    # 2) NumPy's global generator (used by e.g. np.random.rand).
    np.random.seed(seed)

    # 3) PyTorch's CPU generator: controls weight initialisation, torch.randn, the
    #    DataLoader's shuffling order and torchvision's random augmentations.
    torch.manual_seed(seed)

    # 4) PyTorch's generator on every GPU. Safe to call even if there is no GPU.
    torch.cuda.manual_seed_all(seed)

    # cuDNN is NVIDIA's library of fast GPU routines. By default it may pick a
    # different (but equally correct) algorithm each run, based on a speed test
    # ("benchmark"), and some of its algorithms are non-deterministic. These two
    # lines ask it to always choose the same, deterministic algorithm. The cost is
    # a little speed; the benefit is repeatable results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Not done here, on purpose: torch.use_deterministic_algorithms(True) is even
    # stricter, but it makes some GPU operations raise errors unless extra
    # environment variables are set. The two flags above are the usual compromise.


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python `random` inside each DataLoader worker process.

    The DataLoader can start several background processes ("workers") to load
    images in parallel. PyTorch seeds each worker's TORCH generator for us, but
    not its NumPy or Python generators. This function, passed to the DataLoader
    as `worker_init_fn`, fixes that, so augmentations stay reproducible.
    """
    # torch.initial_seed() inside a worker is already a different, deterministic
    # number for each worker (base seed + worker id). We reuse it for NumPy and
    # Python. `% 2**32` keeps it in the range NumPy accepts (`**` means "power").
    worker_seed = torch.initial_seed() % 2**32
    # Seed NumPy's generator in this worker
    np.random.seed(worker_seed)
    # Seed Python's generator in this worker
    random.seed(worker_seed)


class AverageMeter:
    """Keeps a running average of a number, e.g. the mean loss over an epoch.

    Usage:
        meter = AverageMeter()
        meter.update(0.9, n=32)   # a batch of 32 images had mean loss 0.9
        meter.update(0.7, n=16)   # a batch of 16 images had mean loss 0.7
        meter.avg                 # (0.9*32 + 0.7*16) / 48 = 0.8333...
    """

    # PYTHON CONCEPT: `__init__` is the constructor (Java: `public AverageMeter()`).
    # Python calls it automatically when you write `AverageMeter()`. Names with two
    # underscores on each side ("dunder" methods) are hooks Python calls for you in
    # special situations; you rarely call them by hand. `self` is the new object.
    def __init__(self):
        # Start empty. Fields are created simply by assigning to `self.<name>`;
        # unlike Java there is no separate field declaration.
        self.reset()

    def reset(self) -> None:
        # Running total of value * count, so batches of different sizes are weighted
        # correctly (the last batch of an epoch is often smaller).
        self.sum = 0.0
        # How many items have been seen so far.
        self.count = 0
        # Current average; 0.0 until the first update.
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        # Add `n` items whose mean value is `value`.
        self.sum += value * n
        # Count them
        self.count += n
        # Recompute the average; `count` is at least n >= 1 here, so no division by 0
        self.avg = self.sum / self.count


def save_checkpoint(path: str, model, optimizer=None, scheduler=None, **extra) -> None:
    """Save the model's weights (and optionally the optimiser state) to `path`.

    PYTHON CONCEPT: `**extra` in a function DEFINITION collects every additional
    named argument into a dictionary. So
        save_checkpoint("best.pt", model, epoch=7, val_qwk=0.61)
    gives extra = {"epoch": 7, "val_qwk": 0.61}. (The single-star `*args` form
    does the same for unnamed, positional arguments, collecting them into a tuple.)
    It is like Java varargs, but with names attached.
    """
    # Make sure the folder exists. exist_ok=True: no error if it already exists.
    # os.path.dirname("outputs/default/best.pt") -> "outputs/default". It is "" for
    # a bare filename, in which case there is no folder to create.
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    # A checkpoint is just a dictionary. `model.state_dict()` returns an ordered
    # dictionary from parameter name (e.g. "blocks.0.norm.weight") to its tensor.
    state = {"model": model.state_dict()}

    # The optimiser has its own state (AdamW keeps running averages of every
    # gradient). Saving it lets an interrupted run resume exactly where it stopped.
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()

    # Same idea for the learning rate scheduler (which step of the schedule we are at)
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    # Add the extra named values (epoch number, metrics, config dictionary, ...).
    # `dict.update(other)` copies every key/value of `other` into `state`.
    state.update(extra)

    # torch.save serialises the dictionary (tensors included) into one binary file.
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    """Load a checkpoint written by save_checkpoint() back into `model`.

    Returns the full checkpoint dictionary, so the caller can read e.g. the epoch.
    """
    # map_location decides where the tensors are placed when loaded. Loading onto
    # the CPU first works whether the checkpoint was saved from a GPU or not.
    # weights_only=True tells PyTorch to load only tensors and plain Python values
    # (numbers, strings, lists, dicts). The file format is based on Python's
    # "pickle", which can otherwise run arbitrary code hidden in a malicious file.
    state = torch.load(path, map_location=device, weights_only=True)

    # Copy the stored tensors into the model's parameters. strict=True (the
    # default) raises an error if any name is missing or unexpected, which catches
    # "this checkpoint was trained with a different config" mistakes early.
    model.load_state_dict(state["model"])

    # Restore the optimiser too, if the caller wants to resume training.
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    # And the learning rate schedule position.
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    # Give the whole dictionary back to the caller.
    return state


class Logger:
    """Appends one row of metrics per epoch to a CSV file.

    The CSV is what eval/plots.py later reads to draw the training curves. A CSV
    (rather than only printing) means the numbers survive after the Kaggle
    session ends and can be opened in any spreadsheet program.

    Usage:
        logger = Logger("outputs/default/log.csv")
        logger.log({"epoch": 1, "train_loss": 1.2, "val_loss": 1.1})
    """

    def __init__(self, path: str):
        # Where the CSV lives
        self.path = path
        # The column names, decided by the keys of the first row we log
        self.fieldnames = None

        # Create the folder if needed (same pattern as in save_checkpoint)
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        # Start a fresh file for each run: an old log from a previous run would
        # otherwise be mixed with the new rows. Opening with "w" (write) truncates
        # the file to zero length; we immediately close it again by leaving `with`.
        with open(self.path, "w", newline="", encoding="utf-8"):
            pass  # `pass` is Python's "do nothing" statement (an empty `{}` in Java)

    def log(self, row: dict) -> None:
        # The first row fixes the columns; every later row must use the same keys.
        # `list(row.keys())` gives the keys in the order they were inserted.
        first_row = self.fieldnames is None
        if first_row:
            self.fieldnames = list(row.keys())

        # Open in "a" (append) mode so earlier rows are kept. newline="" is what the
        # csv module asks for, so it does not write blank lines on Windows.
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            # DictWriter writes dictionaries as CSV rows, in `fieldnames` order
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            # Write the column names once, at the top of the file
            if first_row:
                writer.writeheader()
            # Write this epoch's values
            writer.writerow(row)
