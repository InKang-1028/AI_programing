# Evaluate every model folder on the test set and save accuracies to a CSV file.
# Uses the same preprocessing as 03_evaluate_ai_model.py.

from tensorflow.keras.models import load_model  # Load a saved Keras model.
import cv2  # Image operations.
import numpy as np  # Array operations.
import csv  # CSV output.
import os  # File and folder operations.

# Model folders to compare; folders without keras_model.h5 are skipped.
model_paths = ["model_1_c10_m10", "model_2_c20_m20", "model_3_c20_m2", "model_4_c200_m200"]

# Class names must match the test folder names and model labels.
test_dir = "./dataset/test"
classes = ["muffin", "chihuahua"]

# Output CSV path.
csv_path = "./results/accuracy.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

# Load all test images once, preprocessed to (224, 224, 3) in [-1, 1].
images, true_labels = [], []
for true_label in classes:
    folder_path = os.path.join(test_dir, true_label)
    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        image = cv2.imread(os.path.join(folder_path, filename))
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        images.append(np.asarray(image, dtype=np.float32) / 127.5 - 1)
        true_labels.append(true_label)
images = np.stack(images)
true_labels = np.array(true_labels)

rows = []
for model_path in model_paths:
    model_file = os.path.join(".", "models", model_path, "keras_model.h5")
    if not os.path.exists(model_file):
        print(f"[skip] {model_file} not found")
        continue

    model = load_model(model_file, compile=False)

    # Remove each label's single-digit index and surrounding whitespace.
    class_names = [line[2:].strip() for line in open(os.path.join(".", "models", model_path, "labels.txt"))]

    # Predict all test images at once.
    prediction = model.predict(images, verbose=0)
    pred_labels = np.array(class_names)[np.argmax(prediction, axis=1)]
    correct = pred_labels == true_labels

    row = {"model": model_path}
    for cls in classes:
        mask = true_labels == cls
        row[f"{cls}_correct"] = int(correct[mask].sum())
        row[f"{cls}_total"] = int(mask.sum())
        row[f"{cls}_acc(%)"] = round(correct[mask].mean() * 100, 2)
    row["overall_correct"] = int(correct.sum())
    row["overall_total"] = len(correct)
    row["overall_acc(%)"] = round(correct.mean() * 100, 2)
    rows.append(row)
    print(row)

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved: {csv_path}")
