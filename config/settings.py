import cv2

# === YOLO SETTINGS ===
YOLO_INPUT_SIZE = 640
YOLO_INPUT_SHAPE = (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)
LETTERBOX_COLOR = (114, 114, 114)
MODEL_ARCHITECTURE = 'yolo11s-seg.pt'
DATA_YAML = 'datasets/yolo_dataset/data.yaml'
TRAINING_OUTPUT_DIR = 'yolo_training_output'
MODEL_NAME = 'yolo11s-seg_frame_detector'

# === YOUR CHARUCO BOARD SETTINGS ===
CHARUCOBOARD_ROWCOUNT = 5
CHARUCOBOARD_COLCOUNT = 5
SQUARE_LENGTH = 0.04  # 40 mm in meters - VERIFY THIS WITH RULER!
MARKER_LENGTH = 0.03  # 30 mm in meters - VERIFY THIS WITH RULER!
CALIB_IMAGES_DIR = "charuco_images/calibration_set"
CALIB_OUTPUT_FILE = "config/s20plus_calib_640x640_letterbox.yaml"  # New filename
ARUCO_DICT = cv2.aruco.DICT_6X6_250  # ← Verify this matches your board generation

# Minimum requirements (adjusted for 5x5 board and letterboxed images)
MIN_MARKERS_PER_IMAGE = 6   # Lowered slightly due to potential edge cropping
MIN_CORNERS_PER_IMAGE = 6   # Lowered slightly due to potential edge cropping
MIN_VALID_IMAGES = 10       # Same as before