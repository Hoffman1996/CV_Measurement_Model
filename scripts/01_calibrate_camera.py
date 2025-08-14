import cv2
import numpy as np
import os
import glob
import yaml

# === YOUR CHARUCO BOARD SETTINGS ===
CHARUCOBOARD_ROWCOUNT = 5
CHARUCOBOARD_COLCOUNT = 5
SQUARE_LENGTH = 0.04  # 40 mm in meters - VERIFY THIS WITH RULER!
MARKER_LENGTH = 0.03  # 30 mm in meters - VERIFY THIS WITH RULER!
CALIB_IMAGES_DIR = "charuco_images/calibration_set"
OUTPUT_FILE = "config/s20plus_calib.yaml"
ARUCO_DICT = cv2.aruco.DICT_6X6_250  # ← Verify this matches your board generation

# Minimum requirements (adjusted for 5x5 board)
MIN_MARKERS_PER_IMAGE = 8   # Lowered from 10 (max is 16 for 5x5)
MIN_CORNERS_PER_IMAGE = 8   # Lowered from 10 (max is 16 for 5x5)
MIN_VALID_IMAGES = 10       # Increased from 5 for better calibration

print("=== ChArUco Camera Calibration ===")
print(f"Board: {CHARUCOBOARD_COLCOUNT}x{CHARUCOBOARD_ROWCOUNT}")
print(f"Square size: {SQUARE_LENGTH*1000:.1f}mm")
print(f"Marker size: {MARKER_LENGTH*1000:.1f}mm")
print(f"Dictionary: {ARUCO_DICT}")
print("=" * 40)

# === BOARD SETUP (OpenCV 4.8+ uses constructor) ===
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
charuco_board = cv2.aruco.CharucoBoard(
    size=(CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
    squareLength=SQUARE_LENGTH,
    markerLength=MARKER_LENGTH,
    dictionary=aruco_dict
)

# === VERIFY IMAGES DIRECTORY ===
if not os.path.exists(CALIB_IMAGES_DIR):
    print(f"❌ Calibration images directory not found: {CALIB_IMAGES_DIR}")
    print("Please take calibration photos first.")
    exit(1)

# === COLLECT CORNERS FROM ALL CALIBRATION IMAGES ===
all_corners = []
all_ids = []
image_size = None
valid_images = []
skipped_images = []

images = glob.glob(os.path.join(CALIB_IMAGES_DIR, "*.jpg"))
images.extend(glob.glob(os.path.join(CALIB_IMAGES_DIR, "*.png")))
print(f"Found {len(images)} calibration images.")

if len(images) == 0:
    print(f"❌ No images found in {CALIB_IMAGES_DIR}")
    exit(1)

for img_path in images:
    print(f"Processing: {os.path.basename(img_path)}", end=" ")
    
    image = cv2.imread(img_path)
    if image is None:
        print("❌ Failed to load")
        skipped_images.append((img_path, "Failed to load image"))
        continue
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Set image size on first valid image
    if image_size is None:
        image_size = gray.shape[::-1]  # (width, height)
        print(f"\nImage size: {image_size[0]}x{image_size[1]}")
    
    # Verify all images have same size
    current_size = gray.shape[::-1]
    if current_size != image_size:
        print(f"❌ Size mismatch: {current_size} vs {image_size}")
        skipped_images.append((img_path, f"Size mismatch: {current_size} vs {image_size}"))
        continue

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is not None and len(ids) >= MIN_MARKERS_PER_IMAGE:
        # Interpolate ChArUco corners
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=charuco_board
        )
        
        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= MIN_CORNERS_PER_IMAGE:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            valid_images.append(img_path)
            print(f"✅ {len(ids)} markers, {len(charuco_ids)} corners")
        else:
            corners_found = len(charuco_ids) if charuco_ids is not None else 0
            print(f"❌ Not enough corners: {corners_found}")
            skipped_images.append((img_path, f"Only {corners_found} ChArUco corners found"))
    else:
        markers_found = len(ids) if ids is not None else 0
        print(f"❌ Not enough markers: {markers_found}")
        skipped_images.append((img_path, f"Only {markers_found} markers found"))

print("\n" + "=" * 40)
print(f"Valid images: {len(valid_images)}")
print(f"Skipped images: {len(skipped_images)}")

if skipped_images:
    print("\nSkipped images:")
    for img_path, reason in skipped_images:
        print(f"  {os.path.basename(img_path)}: {reason}")

# === CALIBRATE IF ENOUGH VALID IMAGES FOUND ===
if len(all_corners) < MIN_VALID_IMAGES:
    print(f"\n❌ Not enough valid calibration images. Found {len(all_corners)}, need at least {MIN_VALID_IMAGES}.")
    print("\nTips for better calibration images:")
    print("- Ensure the entire ChArUco board is visible")
    print("- Take photos from different angles and distances")
    print("- Ensure good lighting with no shadows on the board")
    print("- Keep the board flat and fully in focus")
    print("- Cover different areas of your camera's field of view")
    exit(1)

print(f"\n✅ Proceeding with calibration using {len(all_corners)} images...")

# Perform calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=charuco_board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

# === EVALUATE CALIBRATION QUALITY ===
print("\n" + "=" * 40)
print("CALIBRATION RESULTS")
print("=" * 40)

print(f"Calibration RMS error: {ret:.4f} pixels")
print("Camera Matrix:")
print(camera_matrix)
print("Distortion Coefficients:")
print(dist_coeffs)

# Calculate reprojection errors for quality assessment
total_error = 0
error_details = []

for i in range(len(all_corners)):
    # Project corners back to image
    projected_corners, _ = cv2.projectPoints(
        charuco_board.getChessboardCorners()[all_ids[i].flatten()],
        rvecs[i], tvecs[i], camera_matrix, dist_coeffs
    )
    
    # Calculate error
    error = cv2.norm(all_corners[i], projected_corners, cv2.NORM_L2) / len(projected_corners)
    total_error += error
    error_details.append((valid_images[i], error))

mean_error = total_error / len(all_corners)
print(f"\nMean reprojection error: {mean_error:.4f} pixels")

# Quality assessment
if mean_error < 0.5:
    quality = "Excellent"
elif mean_error < 1.0:
    quality = "Good"
elif mean_error < 2.0:
    quality = "Acceptable"
else:
    quality = "Poor - consider retaking calibration images"

print(f"Calibration quality: {quality}")

# Show worst images if quality is poor
if mean_error > 1.0:
    print("\nImages with highest reprojection errors:")
    error_details.sort(key=lambda x: x[1], reverse=True)
    for img_path, error in error_details[:5]:
        print(f"  {os.path.basename(img_path)}: {error:.4f} pixels")

# === SAVE TO YAML ===
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
calib_data = {
    'camera_matrix': camera_matrix.tolist(),
    'dist_coeff': dist_coeffs.tolist(),
    'image_width': image_size[0],
    'image_height': image_size[1],
    'square_length_m': SQUARE_LENGTH,
    'marker_length_m': MARKER_LENGTH,
    'charuco_dict': ARUCO_DICT,
    'calibration_rms_error': float(ret),
    'mean_reprojection_error': float(mean_error),
    'num_calibration_images': len(all_corners),
    'calibration_quality': quality
}

with open(OUTPUT_FILE, "w") as f:
    yaml.dump(calib_data, f, default_flow_style=False)

print(f"\n📁 Saved calibration to: {OUTPUT_FILE}")
print("\n✅ Calibration complete!")

# === VERIFICATION RECOMMENDATION ===
print("\n" + "=" * 40)
print("NEXT STEPS")
print("=" * 40)
print("1. Test the calibration with new photos")
print("2. If quality is poor, take more varied calibration images")
print("3. Verify measurements are accurate with known objects")
print(f"4. Use the calibration file in your measurement scripts: {OUTPUT_FILE}")