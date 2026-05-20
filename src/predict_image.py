import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
# ÖNEMLI: bu sira flow_from_directory'nin alfabetik siralamasiyla AYNI olmali.
# Klasor adlari: correct_mask, incorrect_mask, no_mask  -> 0, 1, 2
CLASS_NAMES = ["correct_mask", "incorrect_mask", "no_mask"]

MODEL_PATH = "../model/mask_model.h5"

if len(sys.argv) < 2:
    print("Kullanim: python predict_image.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

img = cv2.imread(image_path)
if img is None:
    print("Resim okunamadi:", image_path)
    sys.exit(1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.astype(np.float32)

# MobileNetV2 preprocess_input: [-1, 1] arasina normalize eder.
# Egitim sirasinda KULLANILAN ile AYNI preprocessing burada da uygulanmali.
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)

model = tf.keras.models.load_model(MODEL_PATH)

predictions = model.predict(img, verbose=0)
class_id = int(np.argmax(predictions[0]))
confidence = float(predictions[0][class_id])

print("Prediction :", CLASS_NAMES[class_id])
print("Confidence :", f"{confidence:.4f}")
print("All probabilities:")
for name, p in zip(CLASS_NAMES, predictions[0]):
    print(f"  {name:>16s} : {p:.4f}")
