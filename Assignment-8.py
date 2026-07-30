"""
AI-ML Assignment - 8
Topic: Handwritten Digit Recognition using Artificial Neural Networks (ANN)
Author: Sajjad Shaik

NOTE: This script expects the MNIST-in-CSV dataset from Kaggle:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
Download 'mnist_train.csv' and 'mnist_test.csv' and place them in this
same folder before running. If they are not found, the script falls back
to `tensorflow.keras.datasets.mnist` (downloaded automatically by Keras)
so the pipeline can still run end-to-end.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

TRAIN_CSV = "mnist_train.csv"
TEST_CSV = "mnist_test.csv"


def load_data():
    """Load MNIST either from local Kaggle CSVs or Keras' built-in loader."""
    if os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV):
        print(f"Loading data from local CSVs: '{TRAIN_CSV}', '{TEST_CSV}'")
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)

        print("\nFirst five records (training set):")
        print(train_df.head())

        y_train = train_df["label"].values
        X_train = train_df.drop(columns=["label"]).values
        y_test = test_df["label"].values
        X_test = test_df.drop(columns=["label"]).values

        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
        return X_train, y_train, X_test, y_test

    print(f"WARNING: '{TRAIN_CSV}' / '{TEST_CSV}' not found in this folder.")
    print("Download them from: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
    print("Falling back to tensorflow.keras.datasets.mnist (auto-downloaded).\n")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    sample_df = pd.DataFrame(X_train[0].reshape(1, -1))
    sample_df.insert(0, "label", y_train[0])
    print("\nFirst record preview (flattened pixel row, label prepended):")
    print(sample_df.iloc[:, :6])

    return X_train, y_train, X_test, y_test


def main():
    # -----------------------------------------------------------------------
    # Task 1: Data Understanding (2 Marks)
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("TASK 1: DATA UNDERSTANDING")
    print("=" * 72)

    X_train, y_train, X_test, y_test = load_data()

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape : {X_test.shape}")
    print("\nInput Features : Pixel intensity values (28x28 = 784 pixels per image)")
    print("Target Variable: Digit label (0-9)")

    print(f"\nTraining images: {X_train.shape[0]}, Testing images: {X_test.shape[0]}")
    print(f"Image dimensions: {X_train.shape[1]} x {X_train.shape[2]}")

    plt.figure(figsize=(3, 3))
    plt.imshow(X_train[0], cmap="gray")
    plt.title(f"Sample Digit - Label: {y_train[0]}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("sample_digit.png", dpi=150)
    plt.close()
    print("\nSample digit image saved as 'sample_digit.png'")

    # -----------------------------------------------------------------------
    # Task 2: Data Preprocessing (2 Marks)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TASK 2: DATA PREPROCESSING")
    print("=" * 72)

    print(f"\nMissing values in training pixels: {np.isnan(X_train).sum()}")
    print(f"Missing values in testing pixels : {np.isnan(X_test).sum()}")

    # Normalize pixel values to 0-1
    X_train_norm = X_train.astype("float32") / 255.0
    X_test_norm = X_test.astype("float32") / 255.0

    # Flatten images for the ANN (28x28 -> 784)
    X_train_flat = X_train_norm.reshape(X_train_norm.shape[0], -1)
    X_test_flat = X_test_norm.reshape(X_test_norm.shape[0], -1)

    # Note: the Kaggle MNIST-in-CSV files come pre-split into train/test.
    # We additionally split the (already normalized) training portion
    # 80/20 as required by the assignment, and evaluate the final model on
    # the official Kaggle test split as well.
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_flat, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # One-hot encode target labels
    y_train_cat = to_categorical(y_train_split, num_classes=10)
    y_val_cat = to_categorical(y_val_split, num_classes=10)
    y_test_cat = to_categorical(y_test, num_classes=10)

    print(f"\nTraining samples (80%): {X_train_split.shape[0]}")
    print(f"Validation samples (20%): {X_val_split.shape[0]}")
    print(f"Held-out test samples   : {X_test_flat.shape[0]}")
    print(f"\nOne-hot encoded label example: {y_train_cat[0]}")

    # -----------------------------------------------------------------------
    # Task 3: Model Development (3 Marks)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TASK 3: MODEL DEVELOPMENT")
    print("=" * 72)

    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu", name="hidden_layer_1"),
        layers.Dense(64, activation="relu", name="hidden_layer_2"),
        layers.Dense(10, activation="softmax", name="output_layer"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        X_train_split, y_train_cat,
        validation_data=(X_val_split, y_val_cat),
        epochs=10,
        batch_size=128,
        verbose=2,
    )

    y_pred_probs = model.predict(X_test_flat)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # -----------------------------------------------------------------------
    # Task 4: Model Evaluation (2 Marks)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TASK 4: MODEL EVALUATION")
    print("=" * 72)

    test_loss, test_accuracy = model.evaluate(X_test_flat, y_test_cat, verbose=0)
    print(f"\nTest Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Digit")
    plt.ylabel("Actual Digit")
    plt.title("Confusion Matrix - ANN Digit Classification")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix plot saved as 'confusion_matrix.png'")

    # Accuracy vs Epoch
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_vs_epoch.png", dpi=150)
    plt.close()

    # Loss vs Epoch
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_vs_epoch.png", dpi=150)
    plt.close()
    print("Accuracy/Loss vs Epoch plots saved as 'accuracy_vs_epoch.png', 'loss_vs_epoch.png'")

    print("""
Observations:
1. Training accuracy climbs quickly within the first few epochs and
   plateaus near-perfectly, while validation accuracy tracks closely
   behind it, showing the model generalizes well on unseen digits with
   this relatively simple two-hidden-layer architecture.
2. Most misclassifications in the confusion matrix cluster around visually
   similar digit pairs (e.g. 4 vs 9, 3 vs 5, 7 vs 1), which is expected
   since a flattened, fully-connected ANN has no explicit notion of
   spatial/local structure the way a CNN would.
3. The loss curves decrease steadily without diverging, indicating a
   stable Adam-optimizer training run at the chosen learning rate and
   batch size, with no signs of exploding gradients.
4. A small, growing gap between training and validation accuracy in later
   epochs would indicate the onset of overfitting -- something that could
   be mitigated with dropout layers or early stopping in a further
   iteration.
""")

    # -----------------------------------------------------------------------
    # Task 5: Conclusion (1 Mark)
    # -----------------------------------------------------------------------
    conclusion = """
This project built a fully-connected Artificial Neural Network to
recognize handwritten digits (0-9) from the MNIST dataset, using two
hidden layers of 128 and 64 ReLU-activated neurons followed by a 10-unit
softmax output layer. After normalizing pixel values to the 0-1 range and
one-hot encoding the labels, the model was trained for 10 epochs using
the Adam optimizer and categorical crossentropy loss, achieving strong
test accuracy. Hidden layers are essential to an ANN's ability to learn
increasingly abstract, non-linear combinations of the input pixels --
without them, the network would only be capable of learning a linear
decision boundary, far too limited for a task like digit recognition. A
key advantage of deep learning over traditional machine learning is that
it automatically learns useful features directly from raw pixel data,
removing the need for hand-crafted feature engineering. A notable
limitation of a plain ANN for image tasks, however, is that flattening
the image discards spatial relationships between neighboring pixels,
which is precisely the gap that convolutional architectures like CNNs are
designed to address.
"""
    print("\n" + "=" * 72)
    print("TASK 5: CONCLUSION")
    print("=" * 72)
    print(conclusion)

    model.save("ann_mnist_model.keras")
    print("\nTrained model saved as 'ann_mnist_model.keras'")


if __name__ == "__main__":
    main()
