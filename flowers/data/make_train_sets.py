"""Build the training sets for the two PA04 research questions from pool/.

RQ1 (class imbalance): roses has 200 / 50 / 20 / 5 images, the other classes keep 200.
RQ2 (label noise):     10% / 30% / 50% of each class is relabeled as another class.

All sets start from the same balanced base set (200 images per class, seed 0), so each
experiment changes exactly one thing. Smaller roses sets are subsets of larger ones, and
lower noise levels are subsets of higher ones.

Usage (from this folder, after download_dataset.py):
    python make_train_sets.py

Result:
    train_sets/<experiment>/<class>/*.jpg   folders to upload to Teachable Machine, one class each
    splits/train_<experiment>.csv           path, true label, training label, noisy flag
"""

import csv
import os
import random
import shutil

CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
BASE_PER_CLASS = 200
SEED = 0

MINORITY_CLASS = "roses"
MINORITY_SIZES = [50, 20, 5]   # RQ1; 200 is the base set
NOISE_RATES = [0.1, 0.3, 0.5]  # RQ2; 0.0 is the base set

HERE = os.path.dirname(os.path.abspath(__file__))
POOL_DIR = os.path.join(HERE, "pool")
OUT_DIR = os.path.join(HERE, "train_sets")
SPLITS_DIR = os.path.join(HERE, "splits")


def sample_base(rng):
    """Pick BASE_PER_CLASS images per class from the pool, in a fixed random order."""
    base = {}
    for cls in CLASSES:
        files = sorted(f for f in os.listdir(os.path.join(POOL_DIR, cls)) if f.lower().endswith(".jpg"))
        rng.shuffle(files)
        base[cls] = files[:BASE_PER_CLASS]
    return base


def balanced_set(base):
    """Rows of (file, true label, training label)."""
    return [(name, cls, cls) for cls in CLASSES for name in base[cls]]


def imbalanced_set(base, n_minority):
    """Keep the first n_minority roses images (nested subsets), all other classes unchanged."""
    rows = []
    for cls in CLASSES:
        names = base[cls][:n_minority] if cls == MINORITY_CLASS else base[cls]
        rows += [(name, cls, cls) for name in names]
    return rows


def noisy_set(base, rate):
    """Relabel the first rate*200 images of each class, spread evenly over the other 4 classes.

    Image i of a class always goes to the same wrong class, so lower noise levels are subsets
    of higher ones. Every class still has exactly 200 training images:
    (200 - n) correct + n wrong ones received from the other classes.
    """
    rows = []
    n_noisy = round(rate * BASE_PER_CLASS)
    for cls in CLASSES:
        others = [c for c in CLASSES if c != cls]
        for i, name in enumerate(base[cls]):
            label = others[i % len(others)] if i < n_noisy else cls
            rows.append((name, cls, label))
    return rows


def write_set(exp_name, rows):
    exp_dir = os.path.join(OUT_DIR, exp_name)
    shutil.rmtree(exp_dir, ignore_errors=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(exp_dir, cls), exist_ok=True)
    for name, true_label, train_label in rows:
        shutil.copy2(os.path.join(POOL_DIR, true_label, name), os.path.join(exp_dir, train_label, name))

    with open(os.path.join(SPLITS_DIR, f"train_{exp_name}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "train_label", "noisy"])
        for name, true_label, train_label in sorted(rows, key=lambda r: (r[2], r[0])):
            writer.writerow([f"{true_label}/{name}", true_label, train_label, int(true_label != train_label)])


def main():
    if not os.path.isdir(POOL_DIR):
        raise SystemExit("pool/ not found; run download_dataset.py first")
    os.makedirs(SPLITS_DIR, exist_ok=True)

    base = sample_base(random.Random(SEED))
    experiments = {"base": balanced_set(base)}
    for n in MINORITY_SIZES:
        experiments[f"imb_{MINORITY_CLASS}{n}"] = imbalanced_set(base, n)
    for rate in NOISE_RATES:
        experiments[f"noise{int(rate * 100)}"] = noisy_set(base, rate)

    header = f"{'experiment':16s}" + "".join(f"{c:>12s}" for c in CLASSES) + f"{'total':>8s}{'noisy':>7s}"
    print(header)
    for exp_name, rows in experiments.items():
        write_set(exp_name, rows)
        counts = [sum(1 for r in rows if r[2] == c) for c in CLASSES]
        noisy = sum(1 for r in rows if r[1] != r[2])
        print(f"{exp_name:16s}" + "".join(f"{n:12d}" for n in counts) + f"{len(rows):8d}{noisy:7d}")
    print(f"\nTraining folders: {OUT_DIR}")


if __name__ == "__main__":
    main()
