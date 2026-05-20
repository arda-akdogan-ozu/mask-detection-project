import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# Dataset yolu
DATASET_PATH = "../dataset"
MODEL_DIR = "../model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Parametreler
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 12        # önce sadece head katmanlarini egit
EPOCHS_FINE_TUNE = 8    # sonra MobileNetV2'nin ust katmanlarini ac
SEED = 42

# ---------------------------------------------------------------------------
# 1) DATA GENERATORS
# ---------------------------------------------------------------------------
# ÖNEMLI: train ve validation icin AYRI generator kullaniyoruz.
# Validation'da augmentation uygulanmamali (sadece preprocess_input).
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED,
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED,
)

print("Class indices:", train_generator.class_indices)
NUM_CLASSES = train_generator.num_classes

# ---------------------------------------------------------------------------
# 2) CLASS WEIGHTS (sinif dengesizligini telafi et)
#    incorrect_mask sinifinda ~yari kadar veri var -> daha fazla agirlik alir
# ---------------------------------------------------------------------------
class_labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(class_labels),
    y=class_labels,
)
class_weights = {i: w for i, w in enumerate(class_weights_array)}
print("Class weights:", class_weights)

# ---------------------------------------------------------------------------
# 3) MODEL: MobileNetV2 transfer learning
# ---------------------------------------------------------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # once feature extractor olarak kullan

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------------------------------------------------------------------------
# 4) CALLBACKS
# ---------------------------------------------------------------------------
checkpoint_path = os.path.join(MODEL_DIR, "mask_model.h5")

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

# ---------------------------------------------------------------------------
# 5) STAGE 1 - head egitimi
# ---------------------------------------------------------------------------
print("\n=== STAGE 1: Head training (frozen base) ===")
history_head = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# 6) STAGE 2 - fine tuning (MobileNetV2'nin ust katmanlarini ac)
# ---------------------------------------------------------------------------
print("\n=== STAGE 2: Fine tuning ===")
base_model.trainable = True

# Sadece son ~30 katmani egit, alttakileri donmus birak
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # cok dusuk lr
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE_TUNE,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# 7) KAYDET
# ---------------------------------------------------------------------------
# ModelCheckpoint en iyi val_accuracy'li modeli zaten kaydetti.
# Yine de son halini de kaydedelim (best weights restore edildigi icin ayni model).
model.save(checkpoint_path)
print(f"\nModel kaydedildi: {checkpoint_path}")
print("Class indices (sira onemli):", train_generator.class_indices)

# ---------------------------------------------------------------------------
# 8) GRAFIKLER
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

def _combine(key):
    return history_head.history[key] + history_ft.history[key]

acc = _combine("accuracy")
val_acc = _combine("val_accuracy")
loss = _combine("loss")
val_loss = _combine("val_loss")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="train_accuracy")
plt.plot(val_acc, label="val_accuracy")
plt.axvline(x=len(history_head.history["accuracy"]) - 1, color="gray",
            linestyle="--", label="fine-tune start")
plt.legend()
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.axvline(x=len(history_head.history["loss"]) - 1, color="gray",
            linestyle="--", label="fine-tune start")
plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
plt.show()
