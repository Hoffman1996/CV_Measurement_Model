import cv2
import numpy as np
import os
import glob
import yaml

# === YOUR CHARUCO BOARD SETTINGS ===
CHARUCOBOARD_ROWCOUNT = 5
CHARUCOBOARD_COLCOUNT = 5
SQUARE_LENGTH = 0.04  # 40 mm in meters
MARKER_LENGTH = 0.03  # 30 mm in meters
CALIB_IMAGES_DIR = "charuco_images/calibration_set"
OUTPUT_FILE = "config/s20plus_calib.yaml"
ARUCO_DICT = cv2.aruco.DICT_6X6_250  # ← Correct for your board

# === BOARD SETUP (OpenCV 4.8+ uses constructor) ===
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
charuco_board = cv2.aruco.CharucoBoard(
    size=(CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
    squareLength=SQUARE_LENGTH,
    markerLength=MARKER_LENGTH,
    dictionary=aruco_dict
)

# === COLLECT CORNERS FROM ALL CALIBRATION IMAGES ===
all_corners = []
all_ids = []
image_size = None

images = glob.glob(os.path.join(CALIB_IMAGES_DIR, "*.jpg"))
print(f"Found {len(images)} calibration images.")

for img_path in images:
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # (width, height)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is not None and len(ids) > 10:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=charuco_board
        )
        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 10:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
        else:
            print(f"Skipping {img_path}: not enough ChArUco corners.")
    else:
        print(f"Skipping {img_path}: not enough markers detected.")

# === CALIBRATE IF ENOUGH VALID IMAGES FOUND ===
if len(all_corners) < 5:
    print("❌ Not enough valid calibration images. Need at least 5.")
    exit(1)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=charuco_board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

# === OUTPUT ===
print("\n✅ Calibration successful.")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# === SAVE TO YAML ===
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
calib_data = {
    'camera_matrix': camera_matrix.tolist(),
    'dist_coeff': dist_coeffs.tolist(),
    'image_width': image_size[0],
    'image_height': image_size[1],
    'square_length_m': SQUARE_LENGTH,
    'marker_length_m': MARKER_LENGTH,
    'charuco_dict': ARUCO_DICT
}
with open(OUTPUT_FILE, "w") as f:
    yaml.dump(calib_data, f)

print(f"\n📁 Saved calibration to: {OUTPUT_FILE}")
