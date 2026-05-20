import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "../model/mask_model.h5"
IMG_SIZE = 224

# flow_from_directory alfabetik sirayla yukler:
# correct_mask=0, incorrect_mask=1, no_mask=2
CLASS_NAMES = ["correct_mask", "incorrect_mask", "no_mask"]

# Bir tahminin gosterilmesi icin minimum guven esigi.
# Bu altindakiler "uncertain" olarak gosterilir, sayaca dahil edilmez.
CONFIDENCE_THRESHOLD = 0.60

model = tf.keras.models.load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera acilamadi.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan goruntu alinamadi.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    counts = {"correct_mask": 0, "incorrect_mask": 0, "no_mask": 0}

    for (x, y, w, h) in faces:
        # Yuz cevresine biraz padding ekle (cene/burun maskenin disinda kalabilir).
        pad = int(0.15 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

        # Egitimle AYNI preprocessing
        face_input = preprocess_input(face_resized)
        face_input = np.expand_dims(face_input, axis=0)

        prediction = model.predict(face_input, verbose=0)[0]
        class_id = int(np.argmax(prediction))
        confidence = float(prediction[class_id])

        if confidence < CONFIDENCE_THRESHOLD:
            label = "uncertain"
            color = (180, 180, 180)
        else:
            label = CLASS_NAMES[class_id]
            counts[label] += 1
            if label == "correct_mask":
                color = (0, 255, 0)
            elif label == "incorrect_mask":
                color = (0, 255, 255)
            else:  # no_mask
                color = (0, 0, 255)

        text = f"{label}: {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, text, (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

    total_faces = len(faces)

    cv2.putText(frame, f"Correct Mask: {counts['correct_mask']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Incorrect Mask: {counts['incorrect_mask']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"No Mask: {counts['no_mask']}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Faces: {total_faces}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Automated Face Mask Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
