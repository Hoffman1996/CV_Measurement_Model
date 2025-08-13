# CV_Model_ChArUco – Real-World Window Frame Measurement using AI & Computer Vision

## Final Year Software Engineering Project  
Afeka Tel Aviv Academic College of Engineering  
Author: Yuval Hoffman & Daniel Loevetski

---

## Important Setup Notes

- **CUDA Support:**  
  Make sure your device has a CUDA-capable NVIDIA GPU and the latest NVIDIA drivers installed.
- **CUDA Toolkit:**  
  Install the CUDA Toolkit (recommended version: CUDA 12.6 for PyTorch compatibility).  
  Add the CUDA `bin` directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`) to your system PATH.
- **PyTorch Installation:**  
  Install PyTorch with CUDA support matching your CUDA version.  
  Example for CUDA 12.6:  
  ```
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```
- **Dataset Paths:**  
  Use **absolute paths** in `datasets/yolo_dataset/data.yaml` for `train`, `val`, and `test` image directories to avoid path resolution issues with Ultralytics YOLO.
- **Run Scripts from Project Root:**  
  Always run Python scripts from the project root directory to ensure relative paths resolve correctly.

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
│   └── calibrate_camera.py
├── charuco_images/
│   └── calibration_set/ # Raw images with calibration board
├── config/
│   └── s20plus_calib.yaml # Camera matrix and distortion coefficients
├── datasets/
│   └── yolo_dataset/
│       ├── train/images/
│       ├── valid/images/
│       ├── test/images/
│       └── data.yaml
├── scripts/
│   ├── 01_calibrate_camera.py
│   ├── 03_train_yolo.py
│   └── 04_predict_on_test.py
└── README.md # This file

---

## Troubleshooting

- **CUDA not detected by PyTorch:**  
  - Check your driver and CUDA Toolkit installation.
  - Make sure you installed the correct PyTorch CUDA build.
  - Restart VS Code and your computer after installation.
- **Dataset path errors:**  
  - Use absolute paths in `data.yaml`.
  - Ensure all image folders exist and contain images.
- **YOLO training fails:**  
  - Check error messages for missing files or incorrect paths.
  - Verify your GPU is detected with:
    ```python
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    ```

---

## Quick Start

1. **Install dependencies:**  
   - Python 3.10+
   - CUDA Toolkit 12.6
   - PyTorch with CUDA 12.6
   - Ultralytics YOLO (`pip install -U ultralytics`)
2. **Prepare your dataset:**  
   - Place images in the correct folders.
   - Update `data.yaml` with absolute paths.
3. **Run camera calibration:**  
   ```
   python scripts/01_calibrate_camera.py
   ```
4. **Train YOLO model:**  
   ```
   python scripts/03_train_yolo.py
   ```
5. **Run inference:**  
   ```
   python scripts/04_predict_on_test.py
   ```
