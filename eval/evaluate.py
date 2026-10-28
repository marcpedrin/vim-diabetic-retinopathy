"""Evaluate a trained checkpoint on the EyePACS test split and on Messidor-2.

Run from the repo root:
    python -m eval.evaluate --config configs/default.yaml
    python -m eval.evaluate --checkpoint outputs/default/best.pt

What this file does:
    1. Loads a checkpoint written by train/train.py and rebuilds the model from
       the config stored INSIDE the checkpoint (so the architecture always
       matches the weights).
    2. Rebuilds the exact same patient-level EyePACS split as training (same
       seed and fractions, also taken from the checkpoint) and evaluates on its
       TEST part, which training never touched.
    3. Evaluates on Messidor-2, the external dataset (skipped with a message if
       it is not available).
    4. Prints a side-by-side metrics table, writes all metrics to a JSON file,
       and writes one CSV per dataset with every image's true and predicted
       grade (eval/plots.py uses those for the figures).

Vim paper correspondence (arXiv 2401.09417):
    None of the architecture. The paper evaluates on the ImageNet validation
    set (Section 4.1); we evaluate in-domain (EyePACS test) and cross-domain
    (Messidor-2).

Input / output:
    Input: a checkpoint (.pt) and the config.
    Output: results.json, predictions_eyepacs_test.csv, predictions_messidor2.csv
            (in the folder of cfg.eval.results_json)

Deliberate differences from the paper:
    Not applicable.

The expected cross-dataset drop, and why (domain shift):
    Expect clearly lower scores on Messidor-2 than on the EyePACS test split,
    even though the task is identical. The model only ever saw EyePACS images,
    and Messidor-2 differs in almost every way except the disease:
      - different CAMERAS (Topcon TRC NW6 non-mydriatic camera, 45 degree field)
        versus the many camera models of the EyePACS screening network, giving
        different colour balance, sharpness, field of view and black borders;
      - a different POPULATION (French hospital patients versus a mostly US
        screening programme), with different pigmentation and disease mix;
      - different IMAGE QUALITY and acquisition protocols, and graders working
        to a different reference standard (adjudicated grades).
    Anything the model learnt that is specific to EyePACS photographs rather
    than to the lesions themselves stops working. This is "domain shift", and
    measuring it is exactly why the task asks for an external test set.
"""

import argparse  # command-line options
import csv  # writing per-image prediction files
import json  # writing the results file
import os  # paths

import torch  # loading the checkpoint, devices
import torch.nn as nn  # the (unweighted) loss used for reporting

from data import build_dataloaders, build_messidor_loader  # the two test sets
from eval.metrics import compute_all_metrics  # from-scratch metrics
from train.engine import validate  # the same no-grad evaluation loop as training
from train.utils import load_checkpoint, set_seed  # checkpoint loading, seeding
from vim import build_model  # rebuild the architecture
from vim.config import config_from_dict, load_config  # configs

# DR grade names, used in the printed table
GRADE_NAMES = ["0 none", "1 mild", "2 moderate", "3 severe", "4 prolif."]


def parse_args(argv=None):
    """Command-line options; tests pass `argv` as a list of strings."""
    parser = argparse.ArgumentParser(description="Evaluate a Vision Mamba checkpoint")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="config with the dataset paths")
    parser.add_argument("--override", action="append", default=[],
                        help="change one config value, e.g. eval.checkpoint=x.pt")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint to evaluate (default: eval.checkpoint)")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda' or 'cpu'")
    return parser.parse_args(argv)


def evaluate_loader(model, loader, device, num_classes):
    """Run the model over one loader; return (metrics dict, labels, predictions)."""
    # Unweighted cross-entropy: for REPORTING the loss we want every image to count
    # equally (the class weights were a training device, not part of the score)
    criterion = nn.CrossEntropyLoss()
    out = validate(model, loader, criterion, device)
    metrics = compute_all_metrics(out["labels"], out["predictions"], num_classes)
    metrics["loss"] = out["loss"]
    metrics["num_images"] = int(len(out["labels"]))
    return metrics, out["labels"], out["predictions"]


def write_predictions_csv(path, dataset, labels, predictions):
    """One row per image: file path, true grade, predicted grade.

    The loaders are NOT shuffled, so the i-th prediction belongs to the i-th
    sample of the dataset; that is what lets us pair them up here.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true", "pred"])
        # zip walks the three sequences side by side
        for (image_path, _), true, pred in zip(dataset.samples, labels, predictions):
            writer.writerow([image_path, int(true), int(pred)])


def print_table(results):
    """Print the main metrics side by side, one column per dataset."""
    # Dataset names, e.g. ["eyepacs_test", "messidor2"]
    names = list(results.keys())

    # Header row. In an f-string, `{text:<22}` left-aligns text in a 22-character
    # column and `{text:>16}` right-aligns it in a 16-character column.
    header = f"{'metric':<22}"
    for name in names:
        header += f"{name:>16}"
    print(header)

    # One row per summary metric
    for key in ("num_images", "accuracy", "qwk", "macro_precision", "macro_recall",
                "macro_f1", "loss"):
        line = f"{key:<22}"
        for name in names:
            value = results[name][key]
            # Whole numbers without decimals, everything else with 4 decimals
            if isinstance(value, int):
                line += f"{value:>16d}"
            else:
                line += f"{value:>16.4f}"
        print(line)

    # Per-class recall: the share of each grade that was found.
    # enumerate(list) gives (index, element) pairs: (0, "0 none"), (1, "1 mild"), ...
    for c, grade in enumerate(GRADE_NAMES):
        line = f"{'recall ' + grade:<22}"
        for name in names:
            line += f"{results[name]['per_class_recall'][c]:>16.4f}"
        print(line)


def main(argv=None):
    """Evaluate; returns the results dictionary (used by the tests)."""
    args = parse_args(argv)
    cfg = load_config(args.config, args.override)
    checkpoint_path = args.checkpoint or cfg.eval.checkpoint

    # ---- Device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ---- Rebuild the model exactly as it was trained ----
    # Read the config saved inside the checkpoint (weights_only: safe loading)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    trained_cfg = config_from_dict(state["config"])
    model = build_model(trained_cfg.model)
    load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    print(f"Loaded {checkpoint_path} (epoch {state.get('epoch')}, "
          f"val QWK {state.get('val_qwk', float('nan')):.4f})")

    # ---- Same split as training ----
    # The split depends on the seed and fractions, so take them from the
    # checkpoint's config; only the dataset PATHS come from the current config
    # (they differ between a laptop and Kaggle). Model sizes too must match.
    cfg.seed = trained_cfg.seed
    cfg.data.val_fraction = trained_cfg.data.val_fraction
    cfg.data.test_fraction = trained_cfg.data.test_fraction
    cfg.model = trained_cfg.model
    set_seed(cfg.seed)

    num_classes = cfg.model.num_classes
    results = {}

    # ---- EyePACS test split (in-domain) ----
    # The smoke-test checkpoint was trained on the first N images only; rebuild
    # the same subset so its "test split" is really unseen data
    max_images = None
    if trained_cfg.train.output_dir.endswith("_smoke"):
        max_images = trained_cfg.train.smoke_test_images
    _, _, test_loader = build_dataloaders(cfg, max_images=max_images)
    metrics, labels, preds = evaluate_loader(model, test_loader, device, num_classes)
    results["eyepacs_test"] = metrics

    # ---- Messidor-2 (cross-domain), EVALUATION ONLY ----
    messidor_loader = None
    if os.path.exists(cfg.data.messidor_labels_csv):
        messidor_loader = build_messidor_loader(cfg)
        m_metrics, m_labels, m_preds = evaluate_loader(model, messidor_loader, device,
                                                       num_classes)
        results["messidor2"] = m_metrics
    else:
        print(f"Messidor-2 labels not found at {cfg.data.messidor_labels_csv}; "
              f"skipping the external evaluation")

    # ---- Report ----
    print_table(results)

    # ---- Save ----
    out_dir = os.path.dirname(cfg.eval.results_json) or "."
    os.makedirs(out_dir, exist_ok=True)
    payload = {"checkpoint": checkpoint_path, "epoch": state.get("epoch"),
               "results": results}
    with open(cfg.eval.results_json, "w", encoding="utf-8") as f:
        # indent=2 makes the JSON human-readable
        json.dump(payload, f, indent=2)
    write_predictions_csv(os.path.join(out_dir, "predictions_eyepacs_test.csv"),
                          test_loader.dataset, labels, preds)
    if messidor_loader is not None:
        write_predictions_csv(os.path.join(out_dir, "predictions_messidor2.csv"),
                              messidor_loader.dataset, m_labels, m_preds)
    print(f"Wrote {cfg.eval.results_json}")
    return results


# Run only when executed directly (python -m eval.evaluate), not when imported
if __name__ == "__main__":
    main()
