"""Synthetic fundus-like images, for tests and the --smoke-test run.

What this file does:
    Writes small fake "retina" JPEGs (an orange disc on a black background,
    with dark red dots whose number grows with the grade) plus label CSVs in
    the exact formats of EyePACS (trainLabels.csv) and Messidor-2
    (messidor_data.csv). This lets the whole pipeline, from reading the CSV to
    saving a checkpoint, run on a laptop with no real data downloaded.

Vim paper correspondence (arXiv 2401.09417):
    None. Testing infrastructure only. NOT used for any reported result.

Output:
    make_synthetic_eyepacs(folder, ...)   -> (image_dir, labels_csv)
    make_synthetic_messidor(folder, ...)  -> (image_dir, labels_csv)

Deliberate differences from the paper:
    Not applicable.
"""

import os  # folders and file paths

import numpy as np  # seeded random numbers
import pandas as pd  # writing the CSV files
from PIL import Image, ImageDraw  # creating and drawing on images


def _draw_fake_fundus(grade, rng, width=200, height=160):
    """Draw one fake fundus image for a given grade; returns a PIL image.

    The image is deliberately wider than tall with a black frame, like real
    fundus photos, so the crop-and-pad preprocessing is genuinely exercised.
    """
    # Black RGB canvas
    image = Image.new("RGB", (width, height), (0, 0, 0))
    # ImageDraw gives drawing commands (ellipses, ...) that paint onto `image`
    draw = ImageDraw.Draw(image)

    # The "retina": an orange-red disc, centred, with a little random colour
    radius = 70
    cx, cy = width // 2, height // 2
    # rng.randint(a, b) gives a random integer in [a, b)
    red = 180 + rng.randint(0, 40)
    green = 80 + rng.randint(0, 30)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                 fill=(red, green, 40))

    # The "lesions": 6 dark red dots per grade, at random spots inside the disc
    for _ in range(6 * grade):
        # Random point within 0.8 of the radius, using a random angle and distance
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, 0.8 * radius)
        x = cx + dist * np.cos(angle)
        y = cy + dist * np.sin(angle)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(90, 10, 10))

    return image


def make_synthetic_eyepacs(folder, num_patients=10, seed=0):
    """Write 2 * num_patients fake images (left + right eye) and trainLabels.csv.

    Returns (image_dir, labels_csv), the two paths the EyePACS config needs.
    """
    # Seeded generator: the same seed writes exactly the same files
    rng = np.random.RandomState(seed)

    # Folder layout mirroring the real data: <folder>/train/*.jpeg + trainLabels.csv
    image_dir = os.path.join(folder, "train")
    os.makedirs(image_dir, exist_ok=True)

    rows = []
    for patient in range(num_patients):
        # Patient ids start at 1, as file names like "1_left"
        patient_id = patient + 1
        # Cycle through the grades so every grade is present even for few patients
        grade = patient % 5
        for eye in ("left", "right"):
            name = f"{patient_id}_{eye}"
            image = _draw_fake_fundus(grade, rng)
            # Save as JPEG, the EyePACS format
            image.save(os.path.join(image_dir, name + ".jpeg"), quality=90)
            # One CSV row as a dictionary: column name -> value
            rows.append({"image": name, "level": grade})

    # A DataFrame from a list of dictionaries: one row per dictionary.
    # index=False: do not write pandas' own row numbers as an extra column.
    labels_csv = os.path.join(folder, "trainLabels.csv")
    pd.DataFrame(rows).to_csv(labels_csv, index=False)

    return image_dir, labels_csv


def make_synthetic_messidor(folder, num_images=10, seed=1):
    """Write fake Messidor-2 style images and messidor_data.csv.

    The CSV mimics the real one: ids carry their extension, and the LAST row is
    "ungradable" (empty grade), so the dataset's skipping logic is exercised.
    Returns (image_dir, labels_csv).
    """
    rng = np.random.RandomState(seed)

    # Folder layout: <folder>/images/*.png + messidor_data.csv
    image_dir = os.path.join(folder, "images")
    os.makedirs(image_dir, exist_ok=True)

    rows = []
    for i in range(num_images):
        grade = i % 5
        image_id = f"synthetic_{i:03d}.png"  # `:03d` pads with zeros: 7 -> "007"
        _draw_fake_fundus(grade, rng).save(os.path.join(image_dir, image_id))
        rows.append({"image_id": image_id, "adjudicated_dr_grade": grade,
                     "adjudicated_gradable": 1})

    # One ungradable image: listed, but with no grade (None is written as empty)
    rows.append({"image_id": "synthetic_ungradable.png",
                 "adjudicated_dr_grade": None, "adjudicated_gradable": 0})

    labels_csv = os.path.join(folder, "messidor_data.csv")
    pd.DataFrame(rows).to_csv(labels_csv, index=False)

    return image_dir, labels_csv
