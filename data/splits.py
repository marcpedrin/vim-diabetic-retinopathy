"""Deterministic, stratified, patient-level split of EyePACS into train/val/test.

What this file does:
    Divides the EyePACS label table into three disjoint parts:
        train (~80%)  the model learns from these
        val   (~10%)  used after each epoch to pick the best checkpoint
        test  (~10%)  touched ONCE, at the very end, to report the final score
    with two guarantees:
      1. Both eyes of a patient always land in the SAME part (no leakage).
      2. Each part has roughly the same mix of DR grades (stratification).

Vim paper correspondence (arXiv 2401.09417):
    None. ImageNet comes with an official train/val split; EyePACS's public
    labels come as a single table, so we must split it ourselves.

Input / output:
    split_by_patient(labels_df, ...) -> (train_df, val_df, test_df)
    DataFrames with the same columns as the input (image, level, patient_id).

Deliberate differences from the paper:
    Not applicable.

Why split by PATIENT, not by image (data leakage):
    A patient's left and right eyes look strikingly alike: same camera session,
    same lighting, same pigmentation, usually a similar DR grade. If the left
    eye were in training and the right eye in the test set, the model could
    score well on the test image partly by RECOGNISING the patient rather than
    by grading the disease. The test score would then be optimistic, and it
    would not tell us how the model does on a NEW patient, which is the only
    thing that matters clinically. This is called data leakage, and it is the
    single most common way a student report gets marked down. Grouping by
    patient id removes it.

Why STRATIFY:
    Grades 3 and 4 are rare (about 2-3% of images each). A purely random split
    could, by bad luck, put very few of them in the validation or test set,
    making their per-class scores noisy. We split each grade separately, so
    every part gets about 80/10/10 of each grade.
"""

import numpy as np  # NumPy's random generator, seeded for a reproducible shuffle


def split_by_patient(labels, val_fraction=0.1, test_fraction=0.1, seed=42):
    """Split a label table into (train, val, test) by patient, stratified by grade.

    Args:
        labels: DataFrame with columns image, level, patient_id
                (from data.eyepacs.read_eyepacs_labels).
        val_fraction, test_fraction: share of PATIENTS for validation and test.
        seed: the same seed always gives the same split.
    Returns:
        (train_df, val_df, test_df), three DataFrames with no patient in common.
    """
    # The fractions must leave something to train on
    assert 0 <= val_fraction and 0 <= test_fraction, "fractions must be >= 0"
    assert val_fraction + test_fraction < 1, "val + test must leave training data"

    # One grade per PATIENT, for stratification: the worse (maximum) of their two
    # eyes, since a patient's disease severity is that of their worse eye.
    # groupby("patient_id")["level"].max() gives a Series (a labelled column)
    # mapping patient id -> highest grade, e.g. {"10": 0, "13": 0, "15": 1, ...}.
    patient_grade = labels.groupby("patient_id")["level"].max()

    # A private, seeded random generator: the shuffle below is identical on every
    # run with the same seed, and it does not disturb any other random numbers.
    rng = np.random.RandomState(seed)

    # The patient ids assigned to each part
    train_ids = []
    val_ids = []
    test_ids = []

    # Split each grade's patients separately (this is the stratification)
    for grade in sorted(patient_grade.unique()):
        # All patients whose (worst-eye) grade is `grade`. Sorting first makes the
        # starting order independent of how pandas happened to order them, so
        # only the seed decides the shuffle.
        ids = sorted(patient_grade[patient_grade == grade].index.tolist())

        # Shuffle this grade's patients in place, reproducibly
        rng.shuffle(ids)

        # How many go to test and val. round() to the nearest whole patient;
        # a grade with very few patients may give 0 to val/test (they then all
        # train), which is fine for tiny synthetic data.
        n_test = int(round(len(ids) * test_fraction))
        n_val = int(round(len(ids) * val_fraction))

        # Slices: the first n_test to test, the next n_val to val, the rest to train.
        # `list += other_list` appends all elements of other_list.
        test_ids += ids[:n_test]
        val_ids += ids[n_test : n_test + n_val]
        train_ids += ids[n_test + n_val :]

    # Turn the id lists into row selections of the original table.
    # labels["patient_id"].isin(ids) is True for every row whose patient is in ids,
    # so BOTH eyes of a patient are selected together. .reset_index(drop=True)
    # renumbers the rows 0, 1, 2, ... in each part.
    train_df = labels[labels["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df = labels[labels["patient_id"].isin(val_ids)].reset_index(drop=True)
    test_df = labels[labels["patient_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df
