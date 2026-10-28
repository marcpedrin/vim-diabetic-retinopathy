"""Cross-check every from-scratch metric in eval/metrics.py against scikit-learn.

#####################################################################################
#   scikit-learn is imported HERE ONLY, in this test file, to PROVE that the        #
#   from-scratch metrics are correct. It is never imported by the library,          #
#   training or evaluation code (vim/, data/, train/, eval/), as the task rules     #
#   require. If sklearn is not installed, these tests are skipped, nothing else.    #
#####################################################################################

What is checked, on several random label sets (including imbalanced ones):
    - confusion matrix          vs sklearn.metrics.confusion_matrix
    - accuracy                  vs sklearn.metrics.accuracy_score
    - per-class precision, recall, F1, support
                                vs sklearn.metrics.precision_recall_fscore_support
    - quadratic weighted kappa  vs sklearn.metrics.cohen_kappa_score
                                   with weights="quadratic"
plus the worked example from the QWK docstring and the "always predict grade 0"
baseline.
"""

import os  # for building file paths
import sys  # for editing the module search path

# Add the project root to Python's search path so `import eval` works
# even when this file is run directly (python tests/test_metrics.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # random labels and np.allclose
import pytest  # pytest.importorskip, pytest.mark.parametrize

# importorskip: import sklearn.metrics, or skip this whole file if it is missing
skm = pytest.importorskip("sklearn.metrics")

from eval.metrics import (  # the from-scratch functions under test
    accuracy,
    compute_all_metrics,
    confusion_matrix,
    precision_recall_f1,
    quadratic_weighted_kappa,
)


def random_labels(seed, n=500, imbalanced=False):
    """Random true and predicted grades 0..4 (seeded, so failures are repeatable)."""
    rng = np.random.RandomState(seed)
    if imbalanced:
        # Roughly the EyePACS grade frequencies
        probs = [0.73, 0.07, 0.15, 0.03, 0.02]
        y_true = rng.choice(5, size=n, p=probs)
    else:
        y_true = rng.randint(0, 5, size=n)
    # Predictions: correct 60% of the time, otherwise a random grade
    y_pred = np.where(rng.rand(n) < 0.6, y_true, rng.randint(0, 5, size=n))
    return y_true, y_pred


# PYTEST CONCEPT: parametrize runs the same test once per listed value, here with
# 6 different (seed, imbalanced) combinations, reported as 6 separate tests.
CASES = [(0, False), (1, False), (2, False), (3, True), (4, True), (5, True)]


@pytest.mark.parametrize("seed,imbalanced", CASES)
def test_confusion_matrix_matches_sklearn(seed, imbalanced):
    y_true, y_pred = random_labels(seed, imbalanced=imbalanced)
    ours = confusion_matrix(y_true, y_pred, num_classes=5)
    theirs = skm.confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize("seed,imbalanced", CASES)
def test_accuracy_matches_sklearn(seed, imbalanced):
    y_true, y_pred = random_labels(seed, imbalanced=imbalanced)
    assert np.allclose(accuracy(y_true, y_pred), skm.accuracy_score(y_true, y_pred))


@pytest.mark.parametrize("seed,imbalanced", CASES)
def test_precision_recall_f1_match_sklearn(seed, imbalanced):
    y_true, y_pred = random_labels(seed, imbalanced=imbalanced)
    ours = precision_recall_f1(y_true, y_pred, num_classes=5)
    # zero_division=0: define 0/0 as 0, the same convention as ours
    theirs = skm.precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    # Compare precision, recall, F1 and support one by one
    for our_values, their_values in zip(ours, theirs):
        assert np.allclose(our_values, their_values)


@pytest.mark.parametrize("seed,imbalanced", CASES)
def test_qwk_matches_sklearn(seed, imbalanced):
    y_true, y_pred = random_labels(seed, imbalanced=imbalanced)
    ours = quadratic_weighted_kappa(y_true, y_pred, num_classes=5)
    theirs = skm.cohen_kappa_score(y_true, y_pred, weights="quadratic")
    assert np.allclose(ours, theirs)


def test_qwk_worked_example_from_docstring():
    # The tiny 3-grade example in quadratic_weighted_kappa's docstring
    y_true = [0, 0, 1, 2]
    assert np.isclose(quadratic_weighted_kappa(y_true, [0, 1, 1, 2], 3), 0.8)
    assert np.isclose(quadratic_weighted_kappa(y_true, [0, 1, 1, 1], 3), 3 / 7)
    assert np.isclose(quadratic_weighted_kappa(y_true, [0, 1, 1, 0], 3), -0.25)


def test_always_grade_zero_baseline():
    # EyePACS-like imbalance: a model that always answers "grade 0" gets about 73%
    # accuracy but a QWK of exactly 0: no agreement beyond chance
    y_true, _ = random_labels(seed=10, n=2000, imbalanced=True)
    y_pred = np.zeros_like(y_true)
    metrics = compute_all_metrics(y_true, y_pred)
    print(f"Always-0 baseline: accuracy {metrics['accuracy']:.3f}, "
          f"QWK {metrics['qwk']:.3f}")
    assert metrics["accuracy"] > 0.65
    assert metrics["qwk"] == 0.0


def test_perfect_predictions():
    # Every metric at its best value
    y_true, _ = random_labels(seed=11)
    metrics = compute_all_metrics(y_true, y_true)
    assert metrics["accuracy"] == 1.0
    assert np.isclose(metrics["qwk"], 1.0)
    assert np.isclose(metrics["macro_f1"], 1.0)


# Allow running as a plain script: python tests/test_metrics.py
if __name__ == "__main__":
    for seed, imbalanced in CASES:
        test_confusion_matrix_matches_sklearn(seed, imbalanced)
        test_accuracy_matches_sklearn(seed, imbalanced)
        test_precision_recall_f1_match_sklearn(seed, imbalanced)
        test_qwk_matches_sklearn(seed, imbalanced)
    test_qwk_worked_example_from_docstring()
    test_always_grade_zero_baseline()
    test_perfect_predictions()
    print("OK")
