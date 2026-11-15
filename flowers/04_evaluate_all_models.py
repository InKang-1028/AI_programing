# Evaluate every model folder on the test set and save accuracies to CSV files.
# Adapted from the EF2039 Lab 2 evaluation code (04_evaluate_all_models.py) for the 5-class flower models.
# Uses the same preprocessing as 03_evaluate_ai_model.py.

from tensorflow.keras.models import load_model  # Load a saved Keras model.
import cv2  # Image operations.
import numpy as np  # Array operations.
import csv  # CSV output.
import os  # File and folder operations.

# Model folders to compare; folders without keras_model.h5 are skipped.
# RQ1 (class imbalance): base, imb_roses50, imb_roses20, imb_roses5
# RQ2 (label noise):     base, noise10, noise30, noise50
model_paths = ["base", "imb_roses50", "imb_roses20", "imb_roses5", "noise10", "noise30", "noise50"]

# Class names must match the test folder names and model labels.
test_dir = "./data/test"
classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# Output CSV paths.
csv_path = "./results/accuracy.csv"
cm_path = "./results/confusion_matrix.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

# Load all test images once, preprocessed like Teachable Machine:
# RGB order, center square crop, (224, 224, 3), scaled to [-1, 1].
images, true_labels = [], []
for true_label in classes:
    folder_path = os.path.join(test_dir, true_label)
    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        image = cv2.imread(os.path.join(folder_path, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        side = min(h, w)
        image = image[(h - side) // 2:(h - side) // 2 + side, (w - side) // 2:(w - side) // 2 + side]
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        images.append(np.asarray(image, dtype=np.float32) / 127.5 - 1)
        true_labels.append(true_label)
images = np.stack(images)
true_labels = np.array(true_labels)

rows, cm_rows = [], []
for model_path in model_paths:
    model_file = os.path.join(".", "models", model_path, "keras_model.h5")
    if not os.path.exists(model_file):
        print(f"[skip] {model_file} not found")
        continue

    model = load_model(model_file, compile=False)

    # Remove each label's single-digit index and surrounding whitespace.
    class_names = [line[2:].strip() for line in open(os.path.join(".", "models", model_path, "labels.txt"))]
    assert sorted(class_names) == sorted(classes), f"{model_path}: TM class names {class_names} must be {classes}"

    # Predict all test images at once.
    prediction = model.predict(images, verbose=0)
    pred_labels = np.array(class_names)[np.argmax(prediction, axis=1)]
    correct = pred_labels == true_labels

    row = {"model": model_path}
    f1_scores = []
    for cls in classes:
        mask = true_labels == cls
        row[f"{cls}_correct"] = int(correct[mask].sum())
        row[f"{cls}_total"] = int(mask.sum())
        row[f"{cls}_acc(%)"] = round(correct[mask].mean() * 100, 2)  # per-class accuracy = recall

        # F1 = harmonic mean of precision and recall for this class.
        tp = int((correct & mask).sum())
        n_pred = int((pred_labels == cls).sum())
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / mask.sum()
        f1_scores.append(2 * precision * recall / (precision + recall) if tp else 0.0)

        # Confusion matrix row: how this class's test images were predicted.
        for pred_cls in classes:
            cm_rows.append({"model": model_path, "true": cls, "pred": pred_cls,
                            "count": int((pred_labels[mask] == pred_cls).sum())})
    row["overall_correct"] = int(correct.sum())
    row["overall_total"] = len(correct)
    row["overall_acc(%)"] = round(correct.mean() * 100, 2)
    row["macro_f1(%)"] = round(np.mean(f1_scores) * 100, 2)
    rows.append(row)
    print(f"{model_path:12s} overall {row['overall_acc(%)']:6.2f}%  macro-F1 {row['macro_f1(%)']:6.2f}%  "
          + "  ".join(f"{cls} {row[f'{cls}_acc(%)']:.0f}%" for cls in classes))

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

with open(cm_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "true", "pred", "count"])
    writer.writeheader()
    writer.writerows(cm_rows)

print(f"\nSaved: {csv_path}, {cm_path}")
