import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "../dataset"
MODEL_PATH = "../model/mask_model.h5"
IMG_SIZE = 224
BATCH_SIZE = 32

# Egitimle AYNI preprocessing
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

test_generator = test_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,  # confusion matrix icin sirali olmali
)

model = tf.keras.models.load_model(MODEL_PATH)

loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")
print("Class indices :", test_generator.class_indices)

# Sinif bazli rapor + confusion matrix
print("\nTahminler hesaplaniyor...")
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# class_indices: {'correct_mask': 0, 'incorrect_mask': 1, 'no_mask': 2}
labels_sorted = sorted(test_generator.class_indices, key=test_generator.class_indices.get)

print("\n=== Classification Report ===")
print(classification_report(
    y_true, y_pred, target_names=labels_sorted, digits=4
))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_true, y_pred)
header = "actual \\ pred  " + "  ".join(f"{n:>15s}" for n in labels_sorted)
print(header)
for i, row in enumerate(cm):
    row_str = "  ".join(f"{v:>15d}" for v in row)
    print(f"{labels_sorted[i]:>14s}  {row_str}")
