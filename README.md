# CV_Model_ChArUco – Real-World Window Frame Measurement using AI & Computer Vision

## Final Year Software Engineering Project  
Afeka Tel Aviv Academic College of Engineering  
Author: Yuval Hoffman & Daniel Loevetski

---

## Overview

This project demonstrates how real-world object measurements (specifically **window frames**) can be extracted from smartphone images using **computer vision**, **YOLOv8 object detection**, and **camera calibration**. The system is designed to support a reactive cloud-based architecture and is targeted at non-technical users.

Key features include:
- Camera calibration using a **ChArUco board**
- Training a **YOLOv8 model** to detect full window frames
- Calculating **real-world dimensions** using the calibration matrix
- Outputting a structured **JSON** with width & height in centimeters
- Integration-ready for GCP-based non-blocking backend

---

## Folder Structure

CV_Model_ChArUco/
├── calibration/ # Optional future GUI or CLI scripts
│ └── calibrate_camera.py
├── charuco_images/
│ └── calibration_set/ # Raw images with calibration board
├── config/
│ └── s20plus_calib.yaml # Camera matrix and distortion coefficients
├── datasets/
│ └── yolo_dataset/ # YOLOv8-labeled dataset (from Roboflow)
│ ├── train/images/
│ ├── train/labels/
│ ├── valid/images/
│ ├── valid/labels/
│ └── data.yaml
├── s20plus_images_with_ChArUco/ # Original photo set for calibration
├── scripts/ # Core project logic
│ ├── 01_calibrate_camera.py # Calibration script (done)
│ ├── 02_generate_labels.py # Labeling via contours + calibration (done)
│ ├── 03_train_yolo.py
│ ├── 04_predict_on_test.py
│ ├── 05_generate_json_output.py
│ └── visualize_yolo_labels.py # Visualize YOLO bbox results
├── venv/ # Python virtual environment
├── requirements.txt
└── README.md # This file