"""Train Vision Mamba on EyePACS. Run from the repo root:

    python -m train.train --config configs/default.yaml
    python -m train.train --config configs/default.yaml --override train.epochs=5
    python -m train.train --smoke-test            # 2 epochs, 100 images, any laptop

What this file does:
    Reads the config, seeds everything, builds the data loaders, the model,
    the weighted loss, the AdamW optimiser and the warmup + cosine learning
    rate schedule, then trains for cfg.train.epochs epochs. After every epoch
    it validates, appends a row to a CSV log, saves `last.pt`, and saves
    `best.pt` whenever the validation QWK improves.

Vim paper correspondence (arXiv 2401.09417, Section 4.1, "Settings"):
    The paper trains Vim on ImageNet with AdamW (momentum 0.9, weight decay
    0.05), initial learning rate 1e-3 with a cosine schedule, batch size 1024,
    for 300 epochs on 8 GPUs, using random cropping, horizontal flipping,
    label smoothing, mixup, random erasing and an exponential moving average
    (EMA) of the weights.

Input / output:
    Input: configs/default.yaml (+ overrides); EyePACS images and labels.
    Output folder (cfg.train.output_dir): best.pt, last.pt, log.csv, config.yaml

Deliberate differences from the paper, and why:
    - 30 epochs instead of 300, batch 32 instead of 1024: we have one Kaggle
      GPU for a limited number of hours, the Python-loop scan is slow, and the
      supervisor said the final accuracy does not matter. 30 epochs is enough
      to show convergence curves and a meaningful comparison.
    - Same lr (1e-3) and weight decay (0.05) as the paper. A 32x smaller batch
      would, by the "linear scaling rule", suggest a smaller lr; the warmup and
      gradient clipping keep 1e-3 stable (lower it via --override if not).
    - No mixup, no random erasing, no random crops, no label smoothing: these
      can delete or blur the tiny lesions that define DR grades (see
      data/transforms.py). Flips, rotation and mild colour jitter instead.
    - No EMA of the weights: it helps squeeze out the last fraction of a
      percent over a long run, which is not our goal; it would also double the
      weights kept in memory.
    - Added: class-weighted loss (EyePACS is imbalanced, ImageNet is not),
      gradient clipping, and mixed precision (a speed optimisation only).
    - Checkpoint selection by validation QWK, the competition metric, instead of
      top-1 accuracy (see eval/metrics.py for why accuracy misleads here).
"""

import argparse  # standard library: reads command-line options like --config
import math  # cos and pi for the cosine schedule
import os  # paths
import time  # measuring how long each epoch takes

import torch  # tensors, optimiser, schedulers
import torch.nn as nn  # the loss function
import yaml  # saving a copy of the config next to the results

from data import build_dataloaders  # EyePACS train/val/test loaders
from data.synthetic import make_synthetic_eyepacs  # fake data for the smoke test
from eval.metrics import compute_all_metrics  # accuracy, QWK, F1 ... from scratch
from vim import build_model, count_parameters  # the model
from vim.config import config_to_dict, load_config  # the YAML config

from .engine import train_one_epoch, validate  # the per-epoch loops
from .utils import Logger, save_checkpoint, set_seed  # seeding, logging, saving


def parse_args(argv=None):
    """Read the command-line options. `argv=None` means "use the real command line";
    tests pass a list of strings instead."""
    # ArgumentParser builds a small command-line interface and a --help message
    parser = argparse.ArgumentParser(description="Train Vision Mamba on EyePACS")
    # Path of the YAML config
    parser.add_argument("--config", default="configs/default.yaml",
                        help="path to the YAML config")
    # action="append": the option may be given several times; values are collected
    # into a list, e.g. --override train.epochs=5 --override train.lr=5e-4
    parser.add_argument("--override", action="append", default=[],
                        help="change one config value, e.g. train.epochs=5")
    # action="store_true": a flag with no value; True if present, False if not
    parser.add_argument("--smoke-test", action="store_true",
                        help="2 epochs on 100 images, to check the pipeline end to end")
    # "auto" picks the GPU if there is one
    parser.add_argument("--device", default="auto", help="'auto', 'cuda' or 'cpu'")
    return parser.parse_args(argv)


def class_weights_from_labels(labels, num_classes):
    """Inverse-frequency class weights for the loss: rare grades count more.

    EyePACS is heavily imbalanced: about 73% grade 0, 7% grade 1, 15% grade 2,
    2.5% grade 3 and 2% grade 4. With a plain loss, the model can get most of
    the reward by predicting grade 0 most of the time, and it barely learns the
    rare, clinically most important severe grades.

    The weight of class c is  N / (K * count_c)  (the "balanced" heuristic):
    a class that is exactly 1/K of the data gets weight 1; grade 0 (73%) gets
    about 0.27; grade 4 (2%) gets about 10. In the loss, each image's error is
    multiplied by its class's weight, so in TOTAL each class contributes about
    equally, however many images it has.
    (A gentler alternative is the square root of these weights, if the rare
    classes start to be over-predicted.)
    """
    # How many training images of each grade: a tensor of K counts
    counts = torch.zeros(num_classes)
    for label in labels:
        counts[label] += 1
    # A grade absent from a tiny (smoke-test) training set would divide by zero;
    # clamp(min=1) treats it as one image. Irrelevant on the real data.
    counts = counts.clamp(min=1)
    # N / (K * count_c) for every class at once
    weights = counts.sum() / (num_classes * counts)
    return weights


def build_optimizer(model, lr, weight_decay):
    """AdamW, with weight decay switched OFF for parameters where it does harm.

    AdamW: Adam (a per-weight adaptive step size, based on running averages of
    each weight's gradient and squared gradient) with "decoupled" weight decay,
    which shrinks every weight by lr * weight_decay each step. The paper uses it
    with weight decay 0.05 (Section 4.1).

    Weight decay pulls a weight towards ZERO, which is a sensible prior for the
    big weight matrices, but not for:
      - biases and LayerNorm scales/shifts (1-D tensors): decaying a LayerNorm
        scale towards 0 would just switch features off;
      - the position embeddings and the class token: learned tables, not
        weights of a mapping;
      - A_log: decaying it towards 0 would drag every A towards -exp(0) = -1,
        erasing the spread of memory lengths set up at initialisation.
    The official Vim/Mamba code excludes these too.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        # Skip anything frozen (nothing is, but this keeps the helper general)
        if not param.requires_grad:
            continue
        # param.ndim is the number of dimensions: 1 for a bias or LayerNorm weight.
        # str.endswith(tuple) is True if the name ends with ANY of the strings.
        if param.ndim <= 1 or name.endswith(("pos_embed", "cls_token", "A_log")):
            no_decay.append(param)
        else:
            decay.append(param)
    # Two "parameter groups" with their own weight decay; lr is shared.
    # betas=(0.9, 0.999) are Adam's defaults; 0.9 is the paper's "momentum 0.9".
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def build_scheduler(optimizer, base_lr, min_lr, warmup_steps, total_steps):
    """Linear warmup, then cosine decay, stepped once per batch.

    Learning rate at step s (one step = one batch):
      warmup (s < warmup_steps):  base_lr * (s + 1) / warmup_steps
          grows linearly from almost 0 to base_lr. Early on, the weights are
          random and the gradients large and erratic; full-size steps then can
          throw the model into a bad region it never recovers from.
      after warmup:  min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
          where progress goes from 0 to 1 over the remaining steps. The cosine
          starts flat (keeps a high lr while there is a lot to learn), falls
          fastest in the middle, and flattens out at min_lr (small, careful
          steps to settle into a good minimum).
    """

    # LambdaLR multiplies base_lr by whatever this function returns for step s,
    # so we return the desired lr DIVIDED by base_lr.
    # PYTHON CONCEPT: a function defined inside another function ("nested") can
    # read the outer function's variables (base_lr, min_lr, ...).
    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        # Fraction of the post-warmup steps completed, capped at 1
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine
        return lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def prepare_smoke_test(cfg):
    """Shrink the config for a quick end-to-end check.

    Uses the real EyePACS CSV if it exists; otherwise writes 100 synthetic
    fundus images (data/synthetic.py), so the smoke test runs on any laptop.
    """
    cfg.train.epochs = cfg.train.smoke_test_epochs
    cfg.train.batch_size = cfg.train.smoke_test_batch_size
    # Write to a separate folder ("outputs/default_smoke"), so a quick check can
    # never overwrite the checkpoints of a real training run
    cfg.train.output_dir = cfg.train.output_dir + "_smoke"
    # A short warmup that fits inside a 2-epoch run
    cfg.train.warmup_epochs = min(cfg.train.warmup_epochs, 1)
    # Load images in the main process: simplest, and avoids slow process start-up
    # on Windows laptops
    cfg.data.num_workers = 0

    if not os.path.exists(cfg.data.eyepacs_labels_csv):
        # 50 patients x 2 eyes = 100 images (the default smoke_test_images)
        folder = os.path.join(cfg.train.output_dir, "synthetic_data")
        num_patients = max(cfg.train.smoke_test_images // 2, 5)
        print(f"[smoke test] {cfg.data.eyepacs_labels_csv} not found; writing "
              f"{2 * num_patients} SYNTHETIC images to {folder}")
        image_dir, labels_csv = make_synthetic_eyepacs(folder, num_patients)
        cfg.data.eyepacs_image_dir = image_dir
        cfg.data.eyepacs_labels_csv = labels_csv
    return cfg


def main(argv=None):
    """Run a full training. Returns a small summary dict (used by the tests)."""
    args = parse_args(argv)

    # ---- Config ----
    cfg = load_config(args.config, args.override)
    if args.smoke_test:
        cfg = prepare_smoke_test(cfg)
    max_images = cfg.train.smoke_test_images if args.smoke_test else None

    # ---- Reproducibility: seed everything BEFORE any random number is drawn ----
    # (weight initialisation, the data split, the shuffling order, augmentation)
    set_seed(cfg.seed)

    # ---- Device ----
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    # AMP is a GPU feature; on CPU it is switched off whatever the config says
    use_amp = cfg.train.amp and device.type == "cuda"
    print(f"Device: {device}  |  mixed precision: {use_amp}")

    # ---- Output folder, with a copy of the exact config used ----
    out_dir = cfg.train.output_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)

    # ---- Data ----
    # The test loader is built (so the split is fixed) but NOT used here: the test
    # set is only touched by eval/evaluate.py, once, at the end.
    train_loader, val_loader, _ = build_dataloaders(cfg, max_images=max_images)
    print(f"Images: train {len(train_loader.dataset)}, "
          f"val {len(val_loader.dataset)}")
    # With very few images, rounding 10% of each grade's patients can give zero
    # validation patients. Stop with a clear message rather than crash later.
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError(
            "The train or validation split is empty: use more images "
            "(train.smoke_test_images) or a larger data.val_fraction"
        )

    # ---- Model ----
    model = build_model(cfg.model).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # ---- Loss: cross-entropy, optionally weighted by inverse class frequency ----
    if cfg.train.class_weights:
        weights = class_weights_from_labels(train_loader.dataset.labels,
                                            cfg.model.num_classes)
        # Round for printing: [0.27, 2.9, 1.3, 8.0, 10.1]-ish on the real data
        print("Class weights:", [round(w, 2) for w in weights.tolist()])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # ---- Optimiser and schedule ----
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        base_lr=cfg.train.lr,
        min_lr=cfg.train.min_lr,
        warmup_steps=cfg.train.warmup_epochs * steps_per_epoch,
        total_steps=cfg.train.epochs * steps_per_epoch,
    )
    # GradScaler for mixed precision; with enabled=False every call is a pass-through
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ---- Training loop ----
    logger = Logger(os.path.join(out_dir, "log.csv"))
    # -infinity, so the first epoch's QWK always counts as an improvement
    best_qwk = float("-inf")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    # range(1, epochs + 1) counts 1, 2, ..., epochs (human-friendly epoch numbers)
    for epoch in range(1, cfg.train.epochs + 1):
        start = time.time()

        # One pass over the training set, updating the weights
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler,
            use_amp=use_amp, grad_clip=cfg.train.grad_clip,
        )

        # One pass over the validation set, no updates
        val_out = validate(model, val_loader, criterion, device, use_amp=use_amp)
        val_metrics = compute_all_metrics(val_out["labels"], val_out["predictions"],
                                          cfg.model.num_classes)

        # One row of the CSV log (read by eval/plots.py for the training curves)
        row = {
            "epoch": epoch,
            # The lr at the END of this epoch (first parameter group)
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["accuracy"],
            "val_loss": val_out["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_qwk": val_metrics["qwk"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": round(time.time() - start, 1),
        }
        logger.log(row)
        print(f"epoch {epoch:3d} | train loss {row['train_loss']:.4f} "
              f"acc {row['train_acc']:.3f} | val loss {row['val_loss']:.4f} "
              f"acc {row['val_acc']:.3f} qwk {row['val_qwk']:.4f} | "
              f"{row['epoch_seconds']}s")

        # What every checkpoint stores besides the weights: enough to rebuild the
        # model and to know where it came from
        extra = {"epoch": epoch, "val_qwk": val_metrics["qwk"],
                 "config": config_to_dict(cfg)}

        # Always keep the latest state, so an interrupted run can be resumed
        save_checkpoint(last_path, model, optimizer, scheduler, **extra)

        # Keep the best model by validation QWK (chosen on VALIDATION, never test)
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            save_checkpoint(best_path, model, **extra)
            print(f"  new best val QWK {best_qwk:.4f} -> saved {best_path}")

    print(f"Done. Best validation QWK: {best_qwk:.4f}")
    return {"best_qwk": best_qwk, "best_path": best_path, "output_dir": out_dir}


# Run main() only when this file is executed (python -m train.train), not when it is
# imported (e.g. by the tests). On Windows this guard is also REQUIRED for
# DataLoader worker processes, which re-import this file when they start.
if __name__ == "__main__":
    main()
