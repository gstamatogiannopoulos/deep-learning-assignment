import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
OUT_DIR = "outputs/mura_custom_cnn"


def collect_images(root_dir):
    rows = []
    skipped = 0
    root_dir = Path(root_dir)

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        try:
            img = Image.open(path)
            img.verify()
        except Exception:
            skipped += 1
            continue

        path_str = str(path).replace("\\", "/")
        label = 1 if "positive" in path_str else 0
        study = str(path.parent).replace("\\", "/")
        rows.append([str(path), label, study])

    print("Skipped bad images:", skipped)

    df = pd.DataFrame(rows, columns=["full_path", "label", "study"])
    return df


def read_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.float32)


def make_dataset(df, training=False):
    ds = tf.data.Dataset.from_tensor_slices((df["full_path"].values, df["label"].values))
    ds = ds.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(2000, seed=SEED)
        ds = ds.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate_study_level(model, df):
    ds = make_dataset(df, training=False)
    probs = model.predict(ds, verbose=0).ravel()

    out = df[["study", "label"]].copy()
    out["prob"] = probs

    study_df = out.groupby("study").agg({"label": "max", "prob": "mean"}).reset_index()
    study_df["pred"] = (study_df["prob"] >= 0.5).astype(int)

    auc = roc_auc_score(study_df["label"], study_df["prob"])
    acc = accuracy_score(study_df["label"], study_df["pred"])
    kappa = cohen_kappa_score(study_df["label"], study_df["pred"])
    return study_df, auc, acc, kappa


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to MURA-v1.1 folder")
args = parser.parse_args()

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading MURA...")
train_dir = Path(args.data_dir) / "train"
valid_dir = Path(args.data_dir) / "valid"

if not train_dir.exists() or not valid_dir.exists():
    raise FileNotFoundError(
        "Could not find train/ and valid/ folders inside the path you gave. "
        "Pass the MURA-v1.1 folder in --data_dir."
    )

train_df = collect_images(train_dir)
valid_df = collect_images(valid_dir)

print("Train images:", len(train_df))
print("Valid images:", len(valid_df))

if len(train_df) == 0 or len(valid_df) == 0:
    raise ValueError("No images were found inside train/ or valid/.")

train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
split_n = int(len(train_df) * 0.9)
train_part = train_df.iloc[:split_n].reset_index(drop=True)
val_part = train_df.iloc[split_n:].reset_index(drop=True)

train_ds = make_dataset(train_part, training=True)
val_ds = make_dataset(val_part, training=False)

neg = (train_part["label"] == 0).sum()
pos = (train_part["label"] == 1).sum()
total = neg + pos
class_weight = {
    0: total / (2.0 * max(neg, 1)),
    1: total / (2.0 * max(pos, 1)),
}
print("Class weight:", class_weight)

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
    ],
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_auc", mode="max"),
    keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_auc", mode="max"),
    keras.callbacks.ModelCheckpoint(
        os.path.join(OUT_DIR, "best_model.keras"),
        save_best_only=True,
        monitor="val_auc",
        mode="max",
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

study_df, study_auc, study_acc, study_kappa = evaluate_study_level(model, valid_df)

metrics = {
    "study_auc": float(study_auc),
    "study_accuracy": float(study_acc),
    "study_kappa": float(study_kappa),
    "valid_studies": int(len(study_df)),
}

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(metrics)

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")
plt.plot(history.history["auc"], label="train auc")
plt.plot(history.history["val_auc"], label="val auc")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"))
plt.close()

fpr, tpr, _ = roc_curve(study_df["label"], study_df["prob"])
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {study_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Study ROC curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "study_roc_curve.png"))
plt.close()

cm = confusion_matrix(study_df["label"], study_df["pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "abnormal"])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, colorbar=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "study_confusion_matrix.png"))
plt.close()

model.save(os.path.join(OUT_DIR, "final_model.keras"))
print("Done. Files saved in", OUT_DIR)
