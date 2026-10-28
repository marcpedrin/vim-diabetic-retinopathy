"""End-to-end smoke test: 2 epochs of training on synthetic data.

What is checked:
    - train/train.py runs from start to finish through its real entry point
      (config -> seed -> data -> model -> loss -> optimiser -> loop -> saving)
    - best.pt and last.pt checkpoint files appear, and best.pt loads back
    - the CSV log has one row per epoch, with finite losses
    - the learning-rate schedule warms up and then decays as intended

A 2-block model keeps this fast on a CPU; the code path is the same as a full
run.
"""

import csv  # reading the training log
import math  # math.isfinite
import os  # for building file paths
import sys  # for editing the module search path

# Add the project root to Python's search path so `import train` works
# even when this file is run directly (python tests/test_train_smoke.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # tensors, a dummy optimiser for the schedule test

from data.synthetic import make_synthetic_eyepacs  # fake fundus images
from train.train import build_scheduler, main  # under test
from train.utils import load_checkpoint  # to check the checkpoint loads
from vim import build_model  # to rebuild the model for loading
from vim.config import config_from_dict  # to rebuild the config from a checkpoint

# The real default config, which the run starts from
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def test_two_epoch_training_run(tmp_path):
    # 50 patients x 2 eyes = 100 synthetic images, the default smoke-test size:
    # 10 patients per grade, so 10% val / 10% test gives 1 patient per grade each
    image_dir, labels_csv = make_synthetic_eyepacs(str(tmp_path / "data"),
                                                   num_patients=50)
    out_dir = tmp_path / "run"

    # The same command a user would type, as a list of strings. .as_posix() writes
    # paths with forward slashes, which YAML reads back as plain strings.
    argv = [
        "--config", CONFIG_PATH,
        "--smoke-test",
        "--device", "cpu",
        "--override", f"data.eyepacs_image_dir={image_dir.replace(os.sep, '/')}",
        "--override", f"data.eyepacs_labels_csv={labels_csv.replace(os.sep, '/')}",
        "--override", f"train.output_dir={out_dir.as_posix()}",
        "--override", "model.depth=2",
    ]
    summary = main(argv)

    # --smoke-test writes to "<output_dir>_smoke", so it can never overwrite a real
    # run's checkpoints. tmp_path / "run_smoke" is that folder.
    out_dir = tmp_path / "run_smoke"
    assert summary["output_dir"].replace(os.sep, "/") == out_dir.as_posix()

    # Both checkpoints and the log were written
    assert os.path.exists(out_dir / "best.pt")
    assert os.path.exists(out_dir / "last.pt")
    assert os.path.exists(out_dir / "log.csv")
    assert os.path.exists(out_dir / "config.yaml")

    # The log has exactly 2 rows (2 epochs) with finite losses.
    # csv.DictReader yields one dictionary per row, keyed by the header names.
    with open(out_dir / "log.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print("Log rows:", rows)
    assert len(rows) == 2
    for row in rows:
        assert math.isfinite(float(row["train_loss"]))
        assert math.isfinite(float(row["val_loss"]))

    # best.pt loads back into a model built from the config stored inside it
    state = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=True)
    cfg = config_from_dict(state["config"])
    model = build_model(cfg.model)
    load_checkpoint(str(out_dir / "best.pt"), model)
    assert cfg.model.depth == 2
    assert summary["best_qwk"] > float("-inf")


def test_scheduler_warms_up_then_decays():
    # A dummy parameter and optimiser, just to drive the schedule
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = build_scheduler(optimizer, base_lr=1e-3, min_lr=1e-5,
                                warmup_steps=10, total_steps=100)

    # Record the learning rate at every step
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # Warmup: starts tiny and rises to the peak at the end of warmup
    assert lrs[0] < 2e-4
    assert abs(lrs[9] - 1e-3) < 1e-9
    # Cosine: never increases after the peak (each lr >= the next one)...
    for step in range(9, 99):
        assert lrs[step] >= lrs[step + 1]
    # ...and ends close to min_lr
    assert lrs[-1] < 2e-5


# Allow running as a plain script: python tests/test_train_smoke.py
if __name__ == "__main__":
    import pathlib  # stands in for pytest's tmp_path
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        test_two_epoch_training_run(pathlib.Path(folder))
    test_scheduler_warms_up_then_decays()
    print("OK")
