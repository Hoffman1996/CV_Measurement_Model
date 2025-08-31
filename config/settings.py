# config/setting.py
import cv2

# === YOLO SETTINGS ===
YOLO_INPUT_SIZE = 1024
YOLO_INPUT_SHAPE = (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)
LETTERBOX_COLOR = (255, 255, 255)  # White to match wall paint color
MODEL_ARCHITECTURE = "yolo11s-obb.pt"
MODEL_ARCHITECTURE_NAME = MODEL_ARCHITECTURE[:-3]
DATA_YAML = "datasets/yolo_dataset/data.yaml"
TRAINING_OUTPUT_DIR = "yolo_training_output"
MODEL_NAME = f"{MODEL_ARCHITECTURE_NAME}_frame_detector"
TRAINING_EPOCHS = 150
TRAINING_BATCH_SIZE = 8

# === YOLO TEST SETTINGS ===
TEST_IMAGES_DIR = "datasets/yolo_dataset/test/images"
TEST_LABELS_DIR = "datasets/yolo_dataset/test/labels"
TEST_PREDICTIONS_OUTPUT_DIR = "test_predictions"
VALID_IMAGES_DIR = "datasets/yolo_dataset/valid/images"
VALID_LABELS_DIR = "datasets/yolo_dataset/valid/labels"
CONFIDENCE_THRESHOLD = 0.8
IOU_THRESHOLD = 0.65


# === CHARUCO BOARD SETTINGS ===
CHARUCOBOARD_ROWCOUNT = 5
CHARUCOBOARD_COLCOUNT = 5
SQUARE_LENGTH = 0.04  # 40 mm in meters
MARKER_LENGTH = 0.03  # 30 mm in meters
CALIB_IMAGES_DIR = "charuco_images/calibration_set"
CALIB_OUTPUT_FILE = "config/s20plus_calib_640x640_letterbox.yaml"
IMAGES_WITH_CHARUCO_DIR = "s20plus_images_with_ChArUco"
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

# Minimum requirements (adjusted for 5x5 board and letterboxed images)
MIN_MARKERS_PER_IMAGE = 6
MIN_CORNERS_PER_IMAGE = 6
MIN_VALID_IMAGES = 10

# other settings
VISUALIZATION_DIR = "visualize_corners"

# Detector parameter values
ADAPTIVE_THRESH_WIN_SIZE_MIN = 3
ADAPTIVE_THRESH_WIN_SIZE_MAX = 23
ADAPTIVE_THRESH_WIN_SIZE_STEP = 10
MIN_MARKER_PERIMETER_RATE = 0.01
MAX_MARKER_PERIMETER_RATE = 2.0
CORNER_REFINEMENT_METHOD = cv2.aruco.CORNER_REFINE_SUBPIX


OUTPUT_JSON_FILE = "detection_results_segmentation_fixed.json"

# === WINDOW MEASUREMENT SETTINGS ===
MEASUREMENT_OUTPUT_DIR = "measurement_results"
MEASUREMENT_ACCURACY_TARGET_MM = 10  # Target accuracy in millimeters

unlabeled_images_dir = "testing_images/unlabeled"
labeled_images_dir = "testing_images/labeled"
