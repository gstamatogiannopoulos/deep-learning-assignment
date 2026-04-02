import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUT_DIR = "outputs/fashion_mnist"
os.makedirs(OUT_DIR, exist_ok=True)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("Loading Fashion-MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train_small = x_train[:-5000]
y_train_small = y_train[:-5000]

print("Train:", x_train_small.shape)
print("Val:", x_val.shape)
print("Test:", x_test.shape)

callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
]

print("Training MLP...")
mlp = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

mlp.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_mlp = mlp.fit(
    x_train_small, y_train_small,
    validation_data=(x_val, y_val),
    epochs=15,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

print("Training CNN...")
cnn = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_cnn = cnn.fit(
    x_train_small, y_train_small,
    validation_data=(x_val, y_val),
    epochs=15,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

print("Evaluating models...")
mlp_test_loss, mlp_test_acc = mlp.evaluate(x_test, y_test, verbose=0)
cnn_test_loss, cnn_test_acc = cnn.evaluate(x_test, y_test, verbose=0)

metrics = {
    "mlp_test_accuracy": float(mlp_test_acc),
    "cnn_test_accuracy": float(cnn_test_acc),
    "mlp_test_loss": float(mlp_test_loss),
    "cnn_test_loss": float(cnn_test_loss),
}

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(metrics)

plt.figure(figsize=(8, 5))
plt.plot(history_mlp.history["accuracy"], label="MLP train")
plt.plot(history_mlp.history["val_accuracy"], label="MLP val")
plt.plot(history_cnn.history["accuracy"], label="CNN train")
plt.plot(history_cnn.history["val_accuracy"], label="CNN val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_accuracy.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_mlp.history["loss"], label="MLP train")
plt.plot(history_mlp.history["val_loss"], label="MLP val")
plt.plot(history_cnn.history["loss"], label="CNN train")
plt.plot(history_cnn.history["val_loss"], label="CNN val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_loss.png"))
plt.close()

cnn_preds = np.argmax(cnn.predict(x_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, cnn_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cnn_confusion_matrix.png"))
plt.close()

print("\nCNN classification report:")
print(classification_report(y_test, cnn_preds, target_names=class_names))

idx = np.random.choice(len(x_test), 16, replace=False)
images = x_test[idx]
true_labels = y_test[idx]
pred_labels = cnn_preds[idx]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray")
    ax.set_title(f"T:{class_names[true_labels[i]]}\nP:{class_names[pred_labels[i]]}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_predictions.png"))
plt.close()

mlp.save(os.path.join(OUT_DIR, "mlp_model.keras"))
cnn.save(os.path.join(OUT_DIR, "cnn_model.keras"))

print("Done. Files saved in", OUT_DIR)
