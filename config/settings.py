import cv2

# === YOLO SETTINGS ===
YOLO_INPUT_SIZE = 1024
YOLO_INPUT_SHAPE = (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)
LETTERBOX_COLOR = (255, 255, 255)  # White to match wall paint color
MODEL_ARCHITECTURE = "yolo11s-seg.pt"
model_name_without_suffix = MODEL_ARCHITECTURE[:-3]
DATA_YAML = "datasets/yolo_dataset/data.yaml"
TRAINING_OUTPUT_DIR = "yolo_training_output"
MODEL_NAME = f"{model_name_without_suffix}_frame_detector"
TRAINING_EPOCHS = 150
TRAINING_BATCH_SIZE = 8
TEST_IMAGES_DIR = "datasets/yolo_dataset/test/images"

# === CHARUCO BOARD SETTINGS ===
CHARUCOBOARD_ROWCOUNT = 5
CHARUCOBOARD_COLCOUNT = 5
SQUARE_LENGTH = 0.04  # 40 mm in meters
MARKER_LENGTH = 0.03  # 30 mm in meters
# CALIB_IMAGES_DIR = "charuco_images/calibration_set"
# CALIB_OUTPUT_FILE = "config/s20plus_calib_640x640_letterbox.yaml"
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
