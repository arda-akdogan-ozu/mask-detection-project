import tensorflow as tf
import numpy as np
import cv2
import sys

IMG_SIZE = 224
CLASS_NAMES = ["correct_mask", "incorrect_mask", "no_mask"]

model = tf.keras.models.load_model("../model/mask_model.h5")

if len(sys.argv) < 2:
    print("Kullanım: python3 predict_image.py image_path")
    sys.exit()

image_path = sys.argv[1]

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if img is None:
    print("Resim okunamadı:", image_path)
    sys.exit()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

predictions = model.predict(img)
class_id = np.argmax(predictions[0])
confidence = predictions[0][class_id]

print("Prediction:", CLASS_NAMES[class_id])
print("Confidence:", confidence)
print("All probabilities:", predictions[0])
