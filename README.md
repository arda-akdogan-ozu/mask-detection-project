# Automated Face Mask Detection System using CNNs

## Project Description

This project is an automated face mask detection system developed using Convolutional Neural Networks (CNNs).

The system detects human faces from an image or real-time webcam input and classifies each detected face into one of the following three classes:

1. `correct_mask`
2. `incorrect_mask`
3. `no_mask`

Unlike basic face mask detection systems that only classify faces as "mask" or "no mask", this project also checks whether the mask is worn correctly. In addition, the system counts the number of people in each category in real time.

---

## Main Goal

The main goal of this project is to build a CNN-based real-time system that can:

- Detect faces from webcam input
- Classify each detected face as correct mask, incorrect mask, or no mask
- Display the prediction label and confidence score
- Count the number of people in each category
- Show total detected faces in real time

This makes the project more realistic and useful for real-world monitoring scenarios such as:

- Schools
- Hospitals
- Airports
- Shopping malls
- Offices
- Public transportation areas

---

## Classes

The system uses three classes:

### 1. correct_mask

This class means the mask is worn properly.

A face belongs to this class if:

- The nose is covered
- The mouth is covered
- The mask is placed correctly on the face

Example meaning:

```text
The person is wearing the mask correctly.
```

---

### 2. incorrect_mask

This class means the mask exists but is not worn properly.

A face belongs to this class if:

- The nose is not covered
- The mask is below the nose
- The mask is on the chin
- The mask does not cover the mouth properly

Example meaning:

```text
The person has a mask, but it is worn incorrectly.
```

---

### 3. no_mask

This class means the person is not wearing a mask.

Example meaning:

```text
The person has no mask on their face.
```

---

## Current Features

The following parts are currently implemented:

- Project folder structure
- Dataset organization into 3 classes
- CNN model training
- Model testing
- Single image prediction
- Real-time webcam detection
- Face detection using OpenCV Haar Cascade
- Classification of each detected face
- Bounding box drawing
- Confidence score display
- Real-time category counting
- Training and validation accuracy graph generation
- GitHub repository setup

---

## Technologies Used

This project uses the following technologies:

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Git / GitHub

---

## Project Structure

The expected project structure is:

```text
mask_detection_project/
│
├── dataset/
│   ├── correct_mask/
│   ├── incorrect_mask/
│   └── no_mask/
│
├── model/
│   └── mask_model.h5
│
├── src/
│   ├── train.py
│   ├── test.py
│   ├── predict_image.py
│   └── realtime_detection.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

Important:

The `dataset/` folder and trained model files are not included in the GitHub repository because they may be too large.

Each team member should download and prepare the dataset locally.

---

## GitHub Repository

Repository URL:

```text
https://github.com/arda-akdogan-ozu/mask-detection-project
```

To clone the project:

```bash
git clone https://github.com/arda-akdogan-ozu/mask-detection-project.git
cd mask-detection-project
```

---

## Installation

After cloning the repository, install the required Python packages:

```bash
python3 -m pip install -r requirements.txt
```

If `pip` is not found, use:

```bash
python3 -m ensurepip --upgrade
python3 -m pip install -r requirements.txt
```

To check if TensorFlow is installed correctly:

```bash
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

---

## requirements.txt

The `requirements.txt` file should contain:

```text
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
```

These libraries are used for:

- TensorFlow / Keras: CNN model training and prediction
- OpenCV: image processing, face detection, webcam usage
- NumPy: numerical operations
- Matplotlib: graph visualization
- scikit-learn: future evaluation metrics such as confusion matrix and F1-score

---

## Dataset Setup

The dataset is not included in this repository.

Each team member must manually download and organize the dataset.

---

## Recommended Dataset

Use a Kaggle dataset that contains these three categories:

- Mask
- Incorrect mask
- No mask

Recommended Kaggle search keyword:

```text
Face mask detector mask not mask incorrect mask
```

A suitable dataset usually has folders similar to:

```text
with_mask
mask_weared_incorrect
without_mask
```

or:

```text
correct_mask
incorrect_mask
no_mask
```

---

## How to Organize the Dataset

Inside the project folder, create a folder named `dataset`.

The final dataset structure must be:

```text
dataset/
├── correct_mask/
├── incorrect_mask/
└── no_mask/
```

If these folders do not exist, create them manually:

```bash
mkdir dataset
mkdir dataset/correct_mask
mkdir dataset/incorrect_mask
mkdir dataset/no_mask
```

Then place the images as follows:

```text
with_mask images              -> dataset/correct_mask/
incorrect_mask images         -> dataset/incorrect_mask/
without_mask / no_mask images -> dataset/no_mask/
```

Final example:

```text
dataset/
├── correct_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── incorrect_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── no_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

---

## Checking Dataset Counts

After placing the images, check how many images exist in each folder:

```bash
find dataset/correct_mask -type f | wc -l
find dataset/incorrect_mask -type f | wc -l
find dataset/no_mask -type f | wc -l
```

In our current local dataset, the approximate image counts are:

```text
correct_mask: 690 images
incorrect_mask: 703 images
no_mask: 686 images
```

This is a balanced dataset because all three classes have similar numbers of images.

A balanced dataset is important because if one class has much more data than the others, the model may become biased toward that class.

---

## Training the CNN Model

To train the model, go into the `src` folder:

```bash
cd src
python3 train.py
```

The training script performs the following steps:

1. Loads images from the `dataset/` folder
2. Resizes all images to `224x224`
3. Normalizes pixel values between 0 and 1
4. Applies data augmentation
5. Splits the dataset into training and validation subsets
6. Builds a CNN model
7. Trains the model for 10 epochs
8. Saves the trained model into the `model/` folder

The model is saved as:

```text
model/mask_model.h5
```

Expected training output example:

```text
Found 1664 images belonging to 3 classes.
Found 415 images belonging to 3 classes.
Epoch 1/10
...
Epoch 10/10
Model kaydedildi!
```

---

## CNN Model Architecture

The current model is a custom CNN model.

Model structure:

```text
Input image: 224x224x3

Conv2D layer, 32 filters, ReLU
MaxPooling2D

Conv2D layer, 64 filters, ReLU
MaxPooling2D

Conv2D layer, 128 filters, ReLU
MaxPooling2D

Flatten

Dense layer, 128 neurons, ReLU
Dropout, 0.5

Dense output layer, 3 neurons, Softmax
```

The output layer has 3 neurons because there are 3 classes:

```text
0 -> correct_mask
1 -> incorrect_mask
2 -> no_mask
```

---

## Why Softmax Is Used

The final layer uses softmax activation because this is a multi-class classification problem.

The model gives probabilities for each class.

Example output:

```text
[0.9968, 0.0030, 0.00004]
```

This means:

```text
correct_mask probability   = 99.68%
incorrect_mask probability = 0.30%
no_mask probability        = 0.004%
```

The class with the highest probability becomes the final prediction.

---

## Model Performance

The model was trained for 10 epochs.

Observed performance:

```text
Training Accuracy: approximately 94%
Validation Accuracy: approximately 94%
Test Accuracy: approximately 94.45%
```

The training and validation accuracy curves stay close to each other.

This means:

```text
The model learns successfully and does not show strong overfitting.
```

---

## Testing the Model

To test the trained model, run:

```bash
cd src
python3 test.py
```

Expected output example:

```text
Found 415 images belonging to 3 classes.
Test Loss: 0.1974
Test Accuracy: 0.9445
Class indices: {'correct_mask': 0, 'incorrect_mask': 1, 'no_mask': 2}
```

This means the model achieved about 94.45% accuracy on the validation/test split.

---

## Predicting a Single Image

To predict one image, run:

```bash
cd src
python3 predict_image.py ../dataset/correct_mask/example.jpg
```

Replace `example.jpg` with the actual image name.

To see image names inside a folder:

```bash
ls ../dataset/correct_mask | head
```

Example command:

```bash
python3 predict_image.py ../dataset/correct_mask/0-with-mask.jpg
```

Expected output:

```text
Prediction: correct_mask
Confidence: 0.9968
All probabilities: [0.9968 0.0030 0.00004]
```

Important note:

OpenCV reads images in BGR format, but the model expects RGB format.

Therefore, `predict_image.py` converts images from BGR to RGB before prediction.

---

## Real-Time Webcam Detection

To run real-time detection:

```bash
cd src
python3 realtime_detection.py
```

The system will:

1. Open the webcam
2. Detect faces using OpenCV Haar Cascade
3. Crop each detected face
4. Resize each face to `224x224`
5. Normalize the image
6. Predict the class using the trained CNN model
7. Draw a bounding box around the face
8. Display the predicted label
9. Display the confidence score
10. Count the number of people in each class

To exit the webcam window, press:

```text
q
```

---

## Real-Time Output

The webcam screen displays:

```text
Correct Mask: X
Incorrect Mask: Y
No Mask: Z
Total Faces: N
```

Example:

```text
Correct Mask: 2
Incorrect Mask: 1
No Mask: 3
Total Faces: 6
```

Color coding:

```text
Green  -> Correct Mask
Yellow -> Incorrect Mask
Red    -> No Mask
White  -> Total Faces
```

---

## Current Tested Result

The real-time system was tested with a no-mask face.

Result:

```text
The webcam detected the face successfully.
The model correctly classified it as no_mask.
The counting system worked.
```

Correct mask and incorrect mask tests will be completed later when a physical mask is available.

---

## Common Problems and Solutions

### Problem 1: pip command not found

If this happens:

```text
zsh: command not found: pip
```

Use:

```bash
python3 -m pip install -r requirements.txt
```

---

### Problem 2: Dataset not found

Make sure you are running scripts from the `src` folder:

```bash
cd src
python3 train.py
```

The code expects the dataset at:

```text
../dataset
```

So the folder structure must be:

```text
mask_detection_project/
├── dataset/
└── src/
```

---

### Problem 3: Model file not found

If you see an error like:

```text
No such file or directory: ../model/mask_model.h5
```

Train the model first:

```bash
cd src
python3 train.py
```

This will create:

```text
model/mask_model.h5
```

---

### Problem 4: Webcam does not open

Possible reasons:

- Camera permission is not given
- Another app is using the webcam
- Terminal or IDE does not have camera access

On macOS:

```text
System Settings -> Privacy & Security -> Camera
```

Allow camera access for Terminal, PyCharm, VS Code, or the IDE you are using.

---

### Problem 5: Wrong prediction on single image

Check:

- The image is in the correct class folder
- The image is readable
- BGR to RGB conversion is applied

This project already applies BGR to RGB conversion in `predict_image.py`.

---

### Problem 6: Real-time detection is slow

Possible reasons:

- CPU is weak
- Too many faces are visible
- CNN model is not optimized
- Face detection is slow

Possible future improvements:

- Use MobileNetV2
- Reduce image size
- Measure FPS
- Use a faster face detector
- Use optimized model formats

---

## GitHub Collaboration Guide

For team members:

### First Time Setup

Run:

```bash
git clone https://github.com/arda-akdogan-ozu/mask-detection-project.git
cd mask-detection-project
python3 -m pip install -r requirements.txt
```

Then prepare the dataset manually.

---

### Getting Latest Changes

Before starting work:

```bash
git pull
```

---

### Saving Your Changes

After editing files:

```bash
git add .
git commit -m "Describe your change"
git push
```

Example:

```bash
git add .
git commit -m "Update realtime detection script"
git push
```

---

## Important Git Notes

The following files and folders are ignored by Git:

```text
dataset/
model/*.h5
model/*.keras
__pycache__/
.DS_Store
```

Reason:

- Dataset files can be large
- Trained model files can be large
- Cache files are unnecessary
- macOS `.DS_Store` files should not be uploaded

---

## Completed Work

The following parts have been completed:

- Initial project setup
- Dataset folder structure
- Dataset prepared with 3 classes
- Python dependencies installed
- CNN model created
- CNN model trained
- Model saved
- Model tested
- Single image prediction script implemented
- BGR/RGB prediction issue fixed
- Real-time webcam detection implemented
- Face detection added
- Classification added
- Counting system added
- Training accuracy graph generated
- GitHub repository created
- README documentation written

---

## Future Work

The following improvements are planned:

### 1. Confusion Matrix and Classification Metrics

We plan to add:

- Confusion matrix
- Precision
- Recall
- F1-score

This will help us understand which classes the model confuses the most.

For example, it can show whether the model confuses:

```text
correct_mask vs incorrect_mask
```

---

### 2. More Real-World Testing

We still need to test the system with:

- Correctly worn mask
- Incorrectly worn mask
- No mask
- Multiple people in one frame
- Different lighting conditions
- Different camera distances
- Side faces

---

### 3. Better Face Detection

The current system uses:

```text
OpenCV Haar Cascade
```

Possible improvements:

- MTCNN
- MediaPipe Face Detection
- OpenCV DNN Face Detector

These methods may detect faces better than Haar Cascade.

---

### 4. Better CNN Model

Possible improvements:

- Use MobileNetV2 transfer learning
- Use early stopping
- Save the best validation model
- Tune learning rate
- Tune batch size
- Compare custom CNN with pretrained models

---

### 5. Real-Time Performance Analysis

Since this is a real-time system, we plan to measure:

- FPS
- Average prediction latency
- CPU usage
- Memory usage

This will make the project stronger for the final report.

---

### 6. Final Report and Presentation

The final report should include:

- Introduction
- Problem definition
- Dataset explanation
- Methodology
- CNN architecture
- Training results
- Accuracy/loss graphs
- Real-time detection screenshots
- Limitations
- Future work
- Conclusion

---

## Main Contribution

Most basic face mask detection systems classify faces as:

```text
mask / no mask
```

This project improves that approach by using three classes:

```text
correct_mask / incorrect_mask / no_mask
```

It also adds real-time counting.

Therefore, the system does not only detect masks, but also analyzes mask usage quality and counts people in each category.

---

## Current Status

Current project status:

```text
Core system is working.
```

Working parts:

- Training works
- Testing works
- Single image prediction works
- Real-time webcam detection works
- No-mask detection was tested successfully
- Counting system works

Remaining important tasks:

- Test with correct mask
- Test with incorrect mask
- Test with multiple people
- Add confusion matrix
- Add precision, recall, and F1-score
- Improve face detection
- Improve model architecture
- Prepare final report
