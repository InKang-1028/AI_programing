"""Download TF Flowers and make a fixed test split for the PA04 experiments.

Dataset: TF Flowers (TensorFlow team), 3,670 photos in 5 classes, license CC BY 2.0.
  https://www.tensorflow.org/datasets/catalog/tf_flowers
  https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

Usage (from this folder):
    python download_dataset.py

Result:
    LICENSE.txt                   per-image photographer credits from the archive
    test/<class>/*.jpg            fixed test set, TEST_PER_CLASS images per class
    pool/<class>/*.jpg            all remaining images; training subsets are drawn from here
    splits/test.csv, pool.csv     exact file lists of each split
"""

import csv
import hashlib
import os
import random
import shutil
import tarfile
import urllib.request

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
SHA256 = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"
ARCHIVE = "flower_photos.tgz"
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
TEST_PER_CLASS = 100
SEED = 0

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.chdir(HERE)

    if not os.path.isdir("flower_photos"):
        if not os.path.exists(ARCHIVE):
            print(f"Downloading {URL}")
            urllib.request.urlretrieve(URL, ARCHIVE)
        digest = sha256(ARCHIVE)
        if SHA256 and digest != SHA256:
            raise SystemExit(f"Checksum mismatch: {digest}")
        print(f"sha256 {digest}")
        with tarfile.open(ARCHIVE) as tar:
            tar.extractall()
        os.remove(ARCHIVE)

    # Fixed, seeded split: TEST_PER_CLASS test images per class, the rest go to the pool.
    rng = random.Random(SEED)
    for split in ("test", "pool"):
        shutil.rmtree(split, ignore_errors=True)
    os.makedirs("splits", exist_ok=True)
    rows = {"test": [], "pool": []}
    for cls in CLASSES:
        files = sorted(f for f in os.listdir(os.path.join("flower_photos", cls)) if f.lower().endswith(".jpg"))
        rng.shuffle(files)
        for split, names in (("test", files[:TEST_PER_CLASS]), ("pool", files[TEST_PER_CLASS:])):
            os.makedirs(os.path.join(split, cls), exist_ok=True)
            for name in names:
                shutil.copy2(os.path.join("flower_photos", cls, name), os.path.join(split, cls, name))
                rows[split].append((f"{cls}/{name}", cls))

    for split, items in rows.items():
        with open(os.path.join("splits", f"{split}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            writer.writerows(sorted(items))

    # Keep the credits, then drop the extracted copy (every image is now in test/ or pool/).
    shutil.copy2(os.path.join("flower_photos", "LICENSE.txt"), "LICENSE.txt")
    shutil.rmtree("flower_photos")

    print(f"\n{'class':12s}{'total':>7s}{'test':>7s}{'pool':>7s}")
    for cls in CLASSES:
        n_test = sum(1 for _, c in rows["test"] if c == cls)
        n_pool = sum(1 for _, c in rows["pool"] if c == cls)
        print(f"{cls:12s}{n_test + n_pool:7d}{n_test:7d}{n_pool:7d}")
    print(f"{'total':12s}{len(rows['test']) + len(rows['pool']):7d}{len(rows['test']):7d}{len(rows['pool']):7d}")


if __name__ == "__main__":
    main()
