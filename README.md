# How Training Data Shapes an Image Classifier: Class Imbalance and Label Noise in Teachable Machine

EF2039 Introduction to AI Programming, PA04. Report: [`PA4_Inkang_Jeong.pdf`](PA4_Inkang_Jeong.pdf).

- **RQ1 (class imbalance):** how does shrinking one class (roses: 200 → 50 → 20 → 5 training images) affect accuracy?
- **RQ2 (label noise):** how does accuracy change when 10, 30, or 50% of the training labels are wrong?

## Requirements

- Linux x86_64 (tested on Ubuntu 24.04), Python 3.10, TensorFlow 2.15.1
  (TensorFlow 2.16 and later cannot load Teachable Machine's Keras 2 `.h5` files).
- GPU optional: on 1 CPU core, evaluating all 7 models takes about 36 s and peaks at about 2.5 GB RAM.
- Disk: about 6 GB for the Conda environment, plus 0.9 GB for the dataset.
- Package versions are pinned in `environment.yml` (Conda) and `requirements.txt` (pip).

## 1. Setup

```bash
git clone https://github.com/InKang-1028/AI_programing.git
cd AI_programing
bash setup_env.sh                        # installs Miniconda into .miniconda3/ and creates the env
source .miniconda3/bin/activate tm_flowers
```

With an existing Conda install, run `conda env create -f environment.yml` instead.
Without Conda, run `pip install -r requirements.txt` in a Python 3.10 venv.

## 2. Data

[TF Flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers): 3,670 Flickr photos in 5 classes, CC BY 2.0.
Each image's label is its folder name in the archive.

```bash
cd flowers/data
python download_dataset.py   # download, check sha256, split into test/ and pool/
python make_train_sets.py    # build train_sets/<experiment>/<class>/
cd ..
```

Both scripts use seed 0 (Python `random`). The exact file lists are in `flowers/data/splits/*.csv`.

| Class | daisy | dandelion | roses | sunflowers | tulips |
|---|---|---|---|---|---|
| Total | 633 | 898 | 641 | 699 | 799 |
| **Test** (never used for training) | 100 | 100 | 100 | 100 | 100 |

Training sets: every class has 200 images, except where noted.

| Experiment | `base` | `imb_roses50` | `imb_roses20` | `imb_roses5` | `noise10` | `noise30` | `noise50` |
|---|---|---|---|---|---|---|---|
| Change | none | roses: 50 | roses: 20 | roses: 5 | 10% labels wrong | 30% | 50% |

For the noise sets, the first 10/30/50% of each class are moved to the other four classes in equal shares.

## 3. Models

The trained models are included in `flowers/models/<experiment>/`:
- `keras_model.h5` and `labels.txt`: the Keras export
- `<experiment>.tm`: the TM project
- `*_training_report.png`: TM's training curves

To retrain, open the [Teachable Machine image project](https://teachablemachine.withgoogle.com/train/image) and
make one class per folder, named exactly `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`. Upload
`train_sets/<experiment>/<class>/`, keep the defaults (50 epochs, batch 16, learning rate 0.001), and export as
Keras. TM training is not deterministic, so retrained models may differ slightly.

## 4. Evaluation

```bash
cd flowers
python 04_evaluate_all_models.py   # results/accuracy.csv, results/confusion_matrix.csv
python 05_make_figures.py          # report figure and table, confusion matrices -> results/
python 03_evaluate_ai_model.py     # optional: one model, saves correctly and wrongly classified images
```

- Preprocessing is the same as in TM's export: RGB, center crop, resize to 224×224, scale to [-1, 1].
- Metrics: per-class accuracy (recall), overall accuracy, macro-F1, and confusion matrices.

## 5. Expected results

`results/accuracy.csv` should match this table exactly (test accuracy in %).

| Model | daisy | dandelion | roses | sunflowers | tulips | Accuracy | Macro-F1 |
|---|---|---|---|---|---|---|---|
| `base` | 92 | 84 | 84 | 91 | 82 | 86.6 | 86.6 |
| `imb_roses50` | 90 | 86 | 59 | 91 | 89 | 83.0 | 82.7 |
| `imb_roses20` | 92 | 87 | 36 | 91 | 91 | 79.4 | 78.3 |
| `imb_roses5` | 93 | 86 | 2 | 92 | 90 | 72.6 | 66.5 |
| `noise10` | 83 | 79 | 80 | 87 | 79 | 81.6 | 81.6 |
| `noise30` | 75 | 72 | 70 | 76 | 67 | 72.0 | 72.0 |
| `noise50` | 54 | 53 | 56 | 60 | 48 | 54.2 | 54.2 |
