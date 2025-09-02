# scripts/utils.py
from time import time
import cv2
import os
import numpy as np
import config.settings as settings
from pathlib import Path
from ultralytics import YOLO


def count_images(directory):
    """Count jpg and png images in directory"""
    path = Path(directory)
    if not path.exists():
        return 0
    return len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))


def get_detector_params():
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.adaptiveThreshWinSizeMin = settings.ADAPTIVE_THRESH_WIN_SIZE_MIN
    detector_params.adaptiveThreshWinSizeMax = settings.ADAPTIVE_THRESH_WIN_SIZE_MAX
    detector_params.adaptiveThreshWinSizeStep = settings.ADAPTIVE_THRESH_WIN_SIZE_STEP
    detector_params.minMarkerPerimeterRate = settings.MIN_MARKER_PERIMETER_RATE
    detector_params.maxMarkerPerimeterRate = settings.MAX_MARKER_PERIMETER_RATE
    detector_params.cornerRefinementMethod = settings.CORNER_REFINEMENT_METHOD
    return detector_params


def create_charuco_board():
    return cv2.aruco.CharucoBoard(
        size=(settings.CHARUCOBOARD_COLCOUNT, settings.CHARUCOBOARD_ROWCOUNT),
        squareLength=settings.SQUARE_LENGTH,
        markerLength=settings.MARKER_LENGTH,
        dictionary=settings.ARUCO_DICT,
    )


def detect_charuco_board(
    image, aruco_dict, charuco_board, detector_params, visualize_corners_dir, img_path
):
    """Detect ChArUco board"""

    visualize_corners = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers with improved parameters
    corners, markers_ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=detector_params
    )

    if markers_ids is not None and len(markers_ids) >= settings.MIN_MARKERS_PER_IMAGE:
        # Interpolate ChArUco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=markers_ids,
            image=gray,
            board=charuco_board,
        )
        if (
            retval
            and charuco_corners is not None
            and charuco_ids is not None
            and len(charuco_ids) >= settings.MIN_CORNERS_PER_IMAGE
        ):

            # add print of markers, corners, scale
            cv2.aruco.drawDetectedCornersCharuco(
                visualize_corners, charuco_corners, charuco_ids, (255, 0, 0)
            )
            for i, (pt, cid) in enumerate(zip(charuco_corners, charuco_ids)):
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(visualize_corners, (x, y), 3, (255, 0, 0), -1)
                cv2.putText(
                    visualize_corners,
                    str(int(cid)),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            file_name = os.path.basename(img_path)
            # file_name = f"visualize_corners_{original_image_name}_{len(os.listdir(visualize_corners_dir)) + 1}"

            cv2.imwrite(
                os.path.join(visualize_corners_dir, f"{file_name}"),
                visualize_corners,
            )
        return markers_ids, charuco_corners, charuco_ids

    return None, None, None


def load_model_and_validate_paths(test_dir=None):
    """Load YOLO model and validate all required paths."""
    best_model_path = (
        f"{settings.TRAINING_OUTPUT_DIR}/{settings.MODEL_NAME}/weights/best.pt"
    )

    # Validate model exists
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model not found at: {best_model_path}")

    # Validate test directory exists
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test images directory not found: {test_dir}")

    print(f"Loading model from: {best_model_path}")
    model = YOLO(best_model_path)

    return model, best_model_path


def validate_measurement_paths():
    """Validate paths required for measurement system."""
    best_model_path = (
        f"{settings.TRAINING_OUTPUT_DIR}/{settings.MODEL_NAME}/weights/best.pt"
    )

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"YOLO model not found at: {best_model_path}")

    return best_model_path


def process_single_image(model, img_path, confidence_threshold, iou_threshold):
    """Process a single image and return results."""
    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Failed to load image: {img_path}")
        return None, None, None

    # Time the prediction
    start_time = time()

    # Make prediction
    results = model.predict(
        task="obb",
        source=str(img_path),
        imgsz=settings.YOLO_INPUT_SIZE,
        rect=False,
        conf=confidence_threshold,
        iou=iou_threshold,
        save=False,
        verbose=False,
        max_det=1,  # limit to 3 detections per image
        agnostic_nms=True,  # Force to choose only class 'frame'
    )

    inference_time = time() - start_time
    result = results[0]

    return result, inference_time, image


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


################################################################


def letterbox_yolo_style(
    img, new_shape=settings.YOLO_INPUT_SHAPE, color=settings.LETTERBOX_COLOR
):
    """Apply YOLO letterboxing exactly as used in calibration"""

    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    scale_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute new dimensions
    new_unpad = int(round(shape[1] * scale_ratio)), int(round(shape[0] * scale_ratio))

    # Resize image
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Compute padding
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    # Divide padding into 2 sides
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    img_letterboxed = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return img_letterboxed, scale_ratio, (left, top)
