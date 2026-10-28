"""Report figures: training curves, confusion matrices, example predictions.

Run from the repo root, after training and evaluation:
    python -m eval.plots --config configs/default.yaml

What this file does:
    Reads the CSV training log (train/train.py), the results JSON and the
    per-image prediction CSVs (eval/evaluate.py), and saves PNG figures into
    cfg.eval.figures_dir (report/figures/), where report/main.tex picks them up:
        training_curves.png          report Figure: convergence graphs
        confusion_eyepacs_test.png   report Figure: EyePACS confusion matrix
        confusion_messidor2.png      report Figure: Messidor-2 confusion matrix
        examples_eyepacs_test.png    report Figure: example predictions
        examples_messidor2.png       report Figure: example predictions (external)
    Any input that does not exist yet is skipped with a message.

Vim paper correspondence (arXiv 2401.09417):
    None; these are the deliverables the task asks for in the report.

Input / output:
    CSV / JSON files in, PNG files out. No model is loaded here.

Deliberate differences from the paper:
    Not applicable.

Design choices for the figures:
    - One measure per panel (loss, accuracy, QWK side by side), never two
      y-axes on one plot: two scales on one plot invite false comparisons.
    - Train is always blue and validation always orange, in every panel.
    - The confusion matrices use ONE colour (light to dark blue) for "how
      many", because the cells differ in amount, not in kind.
"""

import argparse  # command-line options
import csv  # reading the log and prediction files
import json  # reading the results file
import os  # paths

import matplotlib

# Use the "Agg" backend: draw into image files without opening any window. This
# must happen BEFORE pyplot is imported, and it makes the script work on Kaggle
# and other machines with no screen.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # the plotting interface (figures, axes, lines)
import numpy as np  # arrays for the confusion matrix
from matplotlib.colors import LinearSegmentedColormap  # a custom one-hue colour ramp
from matplotlib.ticker import MaxNLocator  # whole-number axis ticks

from data.eyepacs import load_rgb_image  # open images for the example grid
from data.transforms import CropToRetina, PadToSquare  # same framing as the model
from vim.config import load_config  # where the inputs and figures live

# Colours (a validated, colour-blind-safe palette): series colours for lines, ink
# colours for all text, and a light-to-dark blue ramp for amounts
TRAIN_COLOUR = "#2a78d6"  # blue
VAL_COLOUR = "#eb6834"  # orange
INK = "#0b0b0b"  # primary text
INK_SECONDARY = "#52514e"  # axis labels, ticks
GRID = "#e4e3df"  # recessive grid lines
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]
)

# Short grade labels for the axes
GRADE_LABELS = ["0", "1", "2", "3", "4"]
GRADE_NAMES = ["none", "mild", "moderate", "severe", "prolif."]


def _style_axis(ax):
    """Quiet axes: no top/right frame, a faint grid behind the data."""
    # Hide the top and right borders of the plot box
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Grey the remaining borders and the tick labels
    ax.spines["left"].set_color(INK_SECONDARY)
    ax.spines["bottom"].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)
    # Horizontal grid lines only, drawn underneath the data
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    # Epochs are whole numbers: never put ticks at 1.2, 1.4, ...
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def read_log(log_csv):
    """Read log.csv into a dict of column name -> list of floats."""
    columns = {}
    with open(log_csv, newline="", encoding="utf-8") as f:
        # DictReader yields one dict per row, keyed by the header names
        for row in csv.DictReader(f):
            for key, value in row.items():
                # setdefault(key, []) returns columns[key], creating an empty list
                # first if the key is new
                columns.setdefault(key, []).append(float(value))
    return columns


def plot_training_curves(log_csv, out_path):
    """REPORT FIGURE: convergence graphs (loss, accuracy, QWK per epoch).

    Three panels sharing the epoch axis: train vs validation loss, train vs
    validation accuracy, and validation QWK (the checkpoint-selection metric).
    """
    log = read_log(log_csv)
    epochs = log["epoch"]

    # One figure, 3 panels in a row. figsize is in inches (width, height).
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    # Panel 1: loss. Note: the training loss is class-WEIGHTED and measured on
    # augmented images, while the validation loss is also weighted (same
    # criterion) but on clean images, so the two are comparable in trend only.
    ax = axes[0]
    ax.plot(epochs, log["train_loss"], color=TRAIN_COLOUR, linewidth=2, label="train")
    ax.plot(epochs, log["val_loss"], color=VAL_COLOUR, linewidth=2, label="validation")
    ax.set_title("Cross-entropy loss", color=INK)
    ax.set_xlabel("epoch", color=INK_SECONDARY)
    ax.legend(frameon=False)
    _style_axis(ax)

    # Panel 2: accuracy
    ax = axes[1]
    ax.plot(epochs, log["train_acc"], color=TRAIN_COLOUR, linewidth=2, label="train")
    ax.plot(epochs, log["val_acc"], color=VAL_COLOUR, linewidth=2, label="validation")
    ax.set_title("Accuracy", color=INK)
    ax.set_xlabel("epoch", color=INK_SECONDARY)
    ax.legend(frameon=False)
    _style_axis(ax)

    # Panel 3: validation QWK (no training QWK is logged), with the best epoch
    # marked, since that is the checkpoint the evaluation uses. A single line
    # needs no legend: the title says what it is (and it keeps the orange
    # "validation" colour of the other panels).
    ax = axes[2]
    ax.plot(epochs, log["val_qwk"], color=VAL_COLOUR, linewidth=2)
    best = int(np.argmax(log["val_qwk"]))
    ax.plot([epochs[best]], [log["val_qwk"][best]], marker="o", markersize=8,
            color=VAL_COLOUR, markeredgecolor="white", markeredgewidth=2)
    best_point = (epochs[best], log["val_qwk"][best])
    ax.annotate(f"best {best_point[1]:.3f}", best_point,
                textcoords="offset points", xytext=(0, 10), ha="center", color=INK)
    ax.set_title("Validation quadratic weighted kappa", color=INK)
    ax.set_xlabel("epoch", color=INK_SECONDARY)
    _style_axis(ax)

    # tight_layout stops titles and labels from overlapping
    fig.tight_layout()
    # dpi = dots per inch: 200 is sharp enough for print
    fig.savefig(out_path, dpi=200)
    # Free the figure's memory (matplotlib keeps figures alive until closed)
    plt.close(fig)


def plot_confusion_matrix(matrix, title, out_path):
    """REPORT FIGURE: confusion matrix for one dataset.

    Cell colour = share of that TRUE grade (each row sums to 100%), so rare
    grades are as readable as grade 0; the cell text gives the raw count too.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    # Row totals, shape (5, 1) so the division below broadcasts across each row.
    # np.maximum(..., 1) avoids dividing by zero for a grade with no images.
    row_totals = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    shares = matrix / row_totals

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    # imshow draws the 5 x 5 grid of coloured cells; vmin/vmax fix the colour
    # scale to 0..1 so figures for different datasets are comparable
    image = ax.imshow(shares, cmap=BLUE_RAMP, vmin=0.0, vmax=1.0)

    # Write "share (count)" in every cell, in white on dark cells, black on light
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text_colour = "white" if shares[i, j] > 0.55 else INK
            ax.text(j, i, f"{shares[i, j]:.0%}\n({int(matrix[i, j])})",
                    ha="center", va="center", fontsize=8, color=text_colour)

    # Axis ticks at the cell centres, labelled with the grades
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(GRADE_LABELS)
    ax.set_yticklabels([f"{g} {n}" for g, n in zip(GRADE_LABELS, GRADE_NAMES)])
    ax.set_xlabel("predicted grade", color=INK_SECONDARY)
    ax.set_ylabel("true grade", color=INK_SECONDARY)
    ax.set_title(title, color=INK)
    # A colour bar explaining what the colours mean
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="share of true grade")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def read_predictions(pred_csv):
    """Read a predictions CSV into a list of (path, true, pred) tuples."""
    rows = []
    with open(pred_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["path"], int(row["true"]), int(row["pred"])))
    return rows


def plot_example_predictions(pred_csv, title, out_path, num_examples=8, seed=0):
    """REPORT FIGURE: a grid of fundus images with true vs predicted grade.

    Picks half correct and half wrong predictions (where available), at random
    but reproducibly (seeded), so the figure shows both successes and failures.
    """
    rows = read_predictions(pred_csv)
    rng = np.random.RandomState(seed)

    # Split into correct and wrong predictions. A list comprehension with an `if`
    # keeps only the elements for which the condition holds (r[1] is the true
    # grade, r[2] the predicted one).
    correct = [r for r in rows if r[1] == r[2]]
    wrong = [r for r in rows if r[1] != r[2]]
    # Shuffle each group reproducibly
    rng.shuffle(correct)
    rng.shuffle(wrong)
    # Up to half the grid from the wrong predictions, the rest from the correct
    n_wrong = min(num_examples // 2, len(wrong))
    n_correct = num_examples - n_wrong
    chosen = wrong[:n_wrong] + correct[:n_correct]
    if not chosen:
        print(f"No predictions in {pred_csv}; skipping {out_path}")
        return

    # Display the images framed as the model sees them (cropped and squared),
    # but at a higher resolution than 128 so lesions remain visible
    crop = CropToRetina()
    pad = PadToSquare()

    cols = 4
    rows_needed = (len(chosen) + cols - 1) // cols  # ceiling division
    # layout="constrained" makes room for each image's two-line title, so titles
    # never overlap the images above them
    fig, axes = plt.subplots(rows_needed, cols, figsize=(2.6 * cols, 3.0 * rows_needed),
                             layout="constrained")
    # np.array(...).reshape(-1) gives a flat list of axes even for a single row
    axes = np.array(axes).reshape(-1)

    for ax, (path, true, pred) in zip(axes, chosen):
        image = pad(crop(load_rgb_image(path, draft_size=512))).resize((256, 256))
        ax.imshow(image)
        # The verdict is spelled out in words, not shown by colour alone
        verdict = "correct" if true == pred else "wrong"
        ax.set_title(f"true {true}, predicted {pred}\n({verdict})", fontsize=9,
                     color=INK)
        ax.axis("off")

    # Hide any unused cells of the grid
    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle(title, color=INK)
    # (No tight_layout here: the constrained layout above already did the spacing)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Make every figure whose inputs exist. Returns the list of files written."""
    parser = argparse.ArgumentParser(description="Make the report figures")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.override)

    fig_dir = cfg.eval.figures_dir
    os.makedirs(fig_dir, exist_ok=True)
    results_dir = os.path.dirname(cfg.eval.results_json) or "."
    written = []

    # Convergence graphs from the training log
    log_csv = os.path.join(cfg.train.output_dir, "log.csv")
    if os.path.exists(log_csv):
        out = os.path.join(fig_dir, "training_curves.png")
        plot_training_curves(log_csv, out)
        written.append(out)
    else:
        print(f"No training log at {log_csv}; skipping training curves")

    # Confusion matrices from the results JSON
    if os.path.exists(cfg.eval.results_json):
        with open(cfg.eval.results_json, encoding="utf-8") as f:
            results = json.load(f)["results"]
        titles = {"eyepacs_test": "EyePACS test split", "messidor2": "Messidor-2"}
        for key, title in titles.items():
            if key in results:
                out = os.path.join(fig_dir, f"confusion_{key}.png")
                plot_confusion_matrix(results[key]["confusion_matrix"], title, out)
                written.append(out)
    else:
        print(f"No results at {cfg.eval.results_json}; skipping confusion matrices")

    # Example prediction grids from the per-image CSVs
    for key, title in (("eyepacs_test", "EyePACS test split"),
                       ("messidor2", "Messidor-2")):
        pred_csv = os.path.join(results_dir, f"predictions_{key}.csv")
        if os.path.exists(pred_csv):
            out = os.path.join(fig_dir, f"examples_{key}.png")
            plot_example_predictions(pred_csv, f"Example predictions: {title}", out)
            written.append(out)

    for path in written:
        print(f"Saved {path}")
    return written


# Run only when executed directly (python -m eval.plots), not when imported
if __name__ == "__main__":
    main()
