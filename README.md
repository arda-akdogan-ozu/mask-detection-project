# Automated Face Mask Detection System using CNNs

## Project Overview
This project detects faces from images or webcam input and classifies each face into:
- correct_mask
- incorrect_mask
- no_mask

The system also counts how many people are in each category in real time.

---

## Installation

Clone the project:
git clone https://github.com/arda-akdogan-ozu/mask-detection-project.git

Go into the folder:
cd mask-detection-project

Install dependencies:
python3 -m pip install -r requirements.txt

---

## Dataset Setup

Create dataset folders:
mkdir dataset
mkdir dataset/correct_mask
mkdir dataset/incorrect_mask
mkdir dataset/no_mask

Download a dataset from Kaggle (search):
Face mask detector mask not mask incorrect mask

Place images like this:

with_mask images → dataset/correct_mask  
incorrect_mask images → dataset/incorrect_mask  
without_mask images → dataset/no_mask  

---

## Train Model

cd src
python3 train.py

Model will be saved to:
model/mask_model.h5

---

## Test Model

cd src
python3 test.py

---

## Predict Single Image

python3 predict_image.py ../dataset/correct_mask/example.jpg

---

## Real-Time Detection

cd src
python3 realtime_detection.py

Press q to exit camera.

---

## Output

Correct Mask: X  
Incorrect Mask: Y  
No Mask: Z  
Total Faces: N  

---

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## Current Status

System is working:
- Training works
- Testing works
- Image prediction works
- Real-time detection works
- Counting works

---

## Future Work

- Confusion matrix
- Precision / Recall / F1
- MobileNetV2
- FPS measurement
- Better face detection

---

## Notes

Dataset is not included in repo.
Each user must download dataset manually.
