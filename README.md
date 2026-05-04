# Automated Face Mask Detection System using CNNs

## Project Overview

This project is an automated face mask detection system developed using Convolutional Neural Networks (CNNs). The system detects human faces from images or real-time camera input and classifies each detected face into one of three categories:

1. Correct Mask
2. Incorrect Mask
3. No Mask

In addition to classification, the system also counts the number of detected faces in each category.

---

## Project Goal

The goal of this project is not only to detect whether a person is wearing a mask, but also to determine whether the mask is worn correctly.

Classes:

- `correct_mask`: mask properly covers nose and mouth  
- `incorrect_mask`: mask worn incorrectly (e.g., nose exposed)  
- `no_mask`: no mask  

---

## Features

- CNN-based image classification
- 3-class mask detection
- Real-time webcam detection
- Face detection using OpenCV
- Bounding box + label + confidence
- Real-time counting system
- Training & validation accuracy analysis

---

## Dataset Structure

```text
dataset/
├── correct_mask/
├── incorrect_mask/
└── no_mask/
