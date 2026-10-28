"""End-to-end test of evaluation and figures on synthetic data.

What is checked:
    - eval/evaluate.py loads a checkpoint from a (tiny) training run, evaluates
      the EyePACS test split and Messidor-2, and writes results.json and the
      per-image prediction CSVs
    - the JSON holds every metric for both datasets
    - eval/plots.py turns those files into the five report figures
"""

import json  # reading results.json
import os  # for building file paths
import sys  # for editing the module search path

# Add the project root to Python's search path so the packages import
# even when this file is run directly (python tests/test_evaluate_and_plots.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.synthetic import make_synthetic_eyepacs, make_synthetic_messidor
from eval import evaluate, plots  # under test
from train import train  # to produce a checkpoint first

# The real default config, which every step starts from
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def posix(path):
    """Forward-slash version of a path, safe inside a YAML override string."""
    return str(path).replace(os.sep, "/")


def test_evaluate_then_plot(tmp_path):
    # Synthetic EyePACS (100 images) and Messidor-2 (10 + 1 ungradable)
    image_dir, labels_csv = make_synthetic_eyepacs(str(tmp_path / "eyepacs"),
                                                   num_patients=50)
    m_dir, m_csv = make_synthetic_messidor(str(tmp_path / "messidor"), num_images=10)

    # Overrides shared by all three steps: data locations, a tiny 2-block model
    run_dir = tmp_path / "run"
    common = [
        "--config", CONFIG_PATH,
        "--override", f"data.eyepacs_image_dir={posix(image_dir)}",
        "--override", f"data.eyepacs_labels_csv={posix(labels_csv)}",
        "--override", f"data.messidor_image_dir={posix(m_dir)}",
        "--override", f"data.messidor_labels_csv={posix(m_csv)}",
        "--override", "data.num_workers=0",
        "--override", f"train.output_dir={posix(run_dir)}",
        "--override", "model.depth=2",
    ]

    # 1) Train (smoke test writes to <output_dir>_smoke)
    summary = train.main(common + ["--smoke-test", "--device", "cpu"])
    smoke_dir = summary["output_dir"]

    # 2) Evaluate the best checkpoint; results go next to it
    eval_paths = [
        "--override", f"eval.checkpoint={posix(os.path.join(smoke_dir, 'best.pt'))}",
        "--override", f"eval.results_json={posix(os.path.join(smoke_dir, 'res.json'))}",
        "--override", f"eval.figures_dir={posix(tmp_path / 'figures')}",
    ]
    results = evaluate.main(common + eval_paths + ["--device", "cpu"])

    # Both datasets were evaluated, with every metric present
    assert set(results.keys()) == {"eyepacs_test", "messidor2"}
    for metrics in results.values():
        for key in ("accuracy", "qwk", "macro_f1", "confusion_matrix", "loss"):
            assert key in metrics
    # Messidor-2: 10 gradable images (the ungradable one was skipped)
    assert results["messidor2"]["num_images"] == 10

    # The JSON and prediction CSVs were written
    with open(os.path.join(smoke_dir, "res.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert "eyepacs_test" in saved["results"]
    assert os.path.exists(os.path.join(smoke_dir, "predictions_eyepacs_test.csv"))
    assert os.path.exists(os.path.join(smoke_dir, "predictions_messidor2.csv"))

    # 3) Figures: the training log lives in the smoke folder
    plot_args = common + eval_paths + ["--override",
                                       f"train.output_dir={posix(smoke_dir)}"]
    written = plots.main(plot_args)
    names = sorted(os.path.basename(p) for p in written)
    print("Figures:", names)
    assert names == [
        "confusion_eyepacs_test.png",
        "confusion_messidor2.png",
        "examples_eyepacs_test.png",
        "examples_messidor2.png",
        "training_curves.png",
    ]


# Allow running as a plain script: python tests/test_evaluate_and_plots.py
if __name__ == "__main__":
    import pathlib  # stands in for pytest's tmp_path
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        test_evaluate_then_plot(pathlib.Path(folder))
    print("OK")
