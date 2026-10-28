"""Classification metrics, all written from scratch with NumPy.

What this file does:
    Given the true grades and the predicted grades of a set of images, compute:
      - the confusion matrix
      - accuracy
      - per-class precision, recall, F1 and support, and their macro averages
      - quadratic weighted kappa (QWK), the official metric of the Kaggle
        EyePACS competition and our main score
    No metric library is used here (the task forbids ready-made metrics outside
    tests); tests/test_metrics.py cross-checks every function against
    scikit-learn.

Vim paper correspondence (arXiv 2401.09417):
    None. The paper reports ImageNet top-1 accuracy; DR grading needs metrics
    that respect class imbalance and the ORDER of the grades.

Input / output:
    y_true, y_pred: 1D integer arrays (or lists) of length n_images, values 0..4
    Outputs are NumPy arrays or plain floats (see each function).

Deliberate differences from the paper:
    Not applicable.

Why plain accuracy is a poor metric on this dataset:
    About 73% of EyePACS images are grade 0 (no DR). A useless model that
    answers "grade 0" for every image, never detecting any disease at all,
    already scores about 73% accuracy. Accuracy also treats every mistake the
    same: calling a proliferative (grade 4) eye "healthy" costs exactly as much
    as calling it "severe" (grade 3). Clinically the first mistake is far
    worse. QWK fixes both problems: that useless model scores a QWK of 0.
"""

import numpy as np  # arrays and vectorised arithmetic


def confusion_matrix(y_true, y_pred, num_classes=5):
    """Count how often each true grade was predicted as each grade.

    Returns an integer array M of shape (num_classes, num_classes) where
    M[i, j] = number of images whose TRUE grade is i and PREDICTED grade is j.
    Rows are the truth, columns the prediction. The diagonal M[i, i] counts the
    correct answers; everything off the diagonal is a mistake, and the further
    from the diagonal, the bigger the mistake (in grades).
    """
    # np.asarray turns lists into arrays (and leaves arrays alone)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Start with all-zero counts; dtype=np.int64 makes them whole numbers
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    # Walk through the images in pairs (true, predicted) and add one to that cell.
    # (A loop is slower than NumPy tricks, but it says exactly what it does.)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def accuracy(y_true, y_pred):
    """Fraction of images whose predicted grade equals the true grade."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # (y_true == y_pred) is an array of True/False; its mean counts True as 1, False
    # as 0, which gives the fraction of correct answers. float() -> Python float.
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, num_classes=5):
    """Per-class precision, recall, F1 and support.

    For one class c (say grade 3):
        precision = of the images we CALLED grade 3, how many really are?
                  = TP / (TP + FP)
        recall    = of the images that REALLY are grade 3, how many did we find?
                  = TP / (TP + FN)
        F1        = 2 * precision * recall / (precision + recall),
                    the harmonic mean: high only when BOTH are high
        support   = how many images really are grade 3
    where TP (true positives)  = really c and predicted c,
          FP (false positives) = predicted c but really another grade,
          FN (false negatives) = really c but predicted another grade.
    When a denominator is zero (e.g. the class was never predicted), the value
    is defined as 0, which matches scikit-learn's default behaviour.

    Returns four arrays of shape (num_classes,): precision, recall, f1, support.
    """
    # All the counts we need are in the confusion matrix
    matrix = confusion_matrix(y_true, y_pred, num_classes)

    # TP for every class at once: the diagonal entries M[c, c]
    tp = np.diag(matrix).astype(np.float64)
    # Column sums = how often each class was PREDICTED (TP + FP)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    # Row sums = how often each class TRULY occurs (TP + FN), i.e. the support
    true_count = matrix.sum(axis=1).astype(np.float64)

    # Precision = TP / (TP + FP), with 0 where nothing was predicted as that class.
    # np.divide(..., out=zeros, where=mask) divides only where mask is True and
    # leaves the zeros elsewhere, avoiding a division-by-zero warning.
    precision = np.divide(tp, predicted_count, out=np.zeros(num_classes),
                          where=predicted_count > 0)
    # Recall = TP / (TP + FN), with 0 where the class never truly occurs
    recall = np.divide(tp, true_count, out=np.zeros(num_classes),
                       where=true_count > 0)
    # F1 = harmonic mean of precision and recall, 0 where both are 0
    p_plus_r = precision + recall
    f1 = np.divide(2 * precision * recall, p_plus_r, out=np.zeros(num_classes),
                   where=p_plus_r > 0)
    # Support as whole numbers
    support = true_count.astype(np.int64)

    return precision, recall, f1, support


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    """Quadratic weighted kappa (QWK): agreement between two graders, beyond chance,
    with bigger disagreements punished quadratically more.

    THE FORMULA, for K classes (K = 5 grades) and N images:

      1. O = the K x K confusion matrix ("observed"): O[i, j] = number of images
         with true grade i and predicted grade j.

      2. W = the K x K quadratic weight matrix ("how bad is each mistake"):
             W[i, j] = (i - j)^2 / (K - 1)^2
         For K = 5 the denominator is 16, so
             off by 0 grades -> 0        (correct: no penalty)
             off by 1 grade  -> 1/16  = 0.0625
             off by 2 grades -> 4/16  = 0.25
             off by 3 grades -> 9/16  = 0.5625
             off by 4 grades -> 16/16 = 1     (grade 0 <-> grade 4: maximum)
         Because of the SQUARE, predicting grade 4 for a healthy (grade 0) eye
         costs 16 times as much as predicting grade 1. That is exactly right for
         an ORDINAL severity scale, where the grades have an order and a
         near-miss is much better than a wild miss. Plain accuracy cannot see
         the difference at all.

      3. E = the confusion matrix we would EXPECT BY CHANCE if the predictions
         were shuffled at random (same overall grade frequencies, no link to the
         images):  E[i, j] = (row total i) * (column total j) / N
         i.e. the outer product of the true-grade histogram and the
         predicted-grade histogram, divided by N.

      4. kappa = 1 - sum(W * O) / sum(W * E)
         sum(W * O) is the total penalty of our actual mistakes; sum(W * E) is
         the total penalty that pure chance would have earned.
           kappa = 1  -> no weighted mistakes at all (perfect)
           kappa = 0  -> no better than chance (e.g. always predicting grade 0)
           kappa < 0  -> worse than chance

    WORKED TINY EXAMPLE (K = 3 grades, so W[i, j] = (i - j)^2 / 4; 4 images):
        true grades       [0, 0, 1, 2]
        predictions       [0, 1, 1, ?]   and we vary only the last prediction ?
        last pred = 2 (correct)       -> QWK =  0.80
        last pred = 1 (off by one)    -> QWK =  0.43
        last pred = 0 (off by two)    -> QWK = -0.25
    Same accuracy (75%) in the last two rows, very different kappa: the
    off-by-two mistake on a grade-2 eye is punished far harder.
    (The full arithmetic for the last row: O has ones at (0,0), (0,1), (1,1),
    (2,0); sum(W * O) = 0.25 + 1 = 1.25. True histogram [2, 1, 1], predicted
    histogram [2, 2, 0], so E = outer([2,1,1], [2,2,0]) / 4 and sum(W * E) = 1.0.
    kappa = 1 - 1.25 / 1.0 = -0.25.)

    Returns a Python float.
    """
    # Step 1: observed counts O, as floats for the arithmetic below. (K, K)
    observed = confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    # N, the number of images
    n = observed.sum()

    # Step 2: the quadratic weight matrix W. (K, K)
    # np.arange(K) = [0, 1, 2, 3, 4]. Reshaping it into a column (K, 1) and a row
    # (1, K) and subtracting broadcasts to the full table of differences i - j.
    grades = np.arange(num_classes)
    diff = grades.reshape(-1, 1) - grades.reshape(1, -1)
    weights = diff.astype(np.float64) ** 2 / (num_classes - 1) ** 2

    # Step 3: expected-by-chance counts E. (K, K)
    true_hist = observed.sum(axis=1)  # how many images truly have each grade, (K,)
    pred_hist = observed.sum(axis=0)  # how often each grade was predicted,     (K,)
    # np.outer(a, b)[i, j] = a[i] * b[j]: every true count times every predicted
    # count, then divided by N
    expected = np.outer(true_hist, pred_hist) / n

    # Step 4: kappa = 1 - sum(W * O) / sum(W * E)
    observed_penalty = np.sum(weights * observed)
    expected_penalty = np.sum(weights * expected)

    # Edge case: if chance itself makes no weighted mistakes (e.g. every true and
    # every predicted grade is the same single class), kappa is 0/0, undefined.
    # There is then no "agreement beyond chance" to measure, so we return 0.0.
    if expected_penalty == 0:
        return 0.0

    return float(1.0 - observed_penalty / expected_penalty)


def compute_all_metrics(y_true, y_pred, num_classes=5):
    """Every metric above, in one dictionary of plain Python values.

    Plain floats and lists (not NumPy arrays) so the result can be written
    straight to a JSON file and into the CSV training log.
    """
    precision, recall, f1, support = precision_recall_f1(y_true, y_pred, num_classes)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred, num_classes),
        # Macro average = plain mean over the 5 classes, so a rare grade counts as
        # much as grade 0. (A "weighted" average would let grade 0 dominate again.)
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        # .tolist() turns NumPy arrays into ordinary Python lists
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes).tolist(),
    }
