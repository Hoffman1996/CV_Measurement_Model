# 01_charuco_calibration_640x640_letterbox.py
import cv2
import numpy as np
import os
import glob
import yaml
import scripts.utils as utils
import config.settings as settings


print("=== ChArUco Camera Calibration for YOLO (640x640 Letterboxed) ===")
print(f"Board: {settings.CHARUCOBOARD_COLCOUNT}x{settings.CHARUCOBOARD_ROWCOUNT}")
print(f"Square size: {settings.SQUARE_LENGTH*1000:.1f}mm")
print(f"Marker size: {settings.MARKER_LENGTH*1000:.1f}mm")
print(f"Dictionary: {settings.ARUCO_DICT}")
print(f"YOLO input size: {settings.YOLO_INPUT_SIZE}x{settings.YOLO_INPUT_SIZE}")
print(f"Letterbox padding color: {settings.LETTERBOX_COLOR}")
print("=" * 50)

# === BOARD SETUP (OpenCV 4.8+ uses constructor) ===
aruco_dict = cv2.aruco.getPredefinedDictionary(settings.ARUCO_DICT)
charuco_board = cv2.aruco.CharucoBoard(
    size=(settings.CHARUCOBOARD_COLCOUNT, settings.CHARUCOBOARD_ROWCOUNT),
    squareLength=settings.SQUARE_LENGTH,
    markerLength=settings.MARKER_LENGTH,
    dictionary=aruco_dict
)

# === VERIFY IMAGES DIRECTORY ===
if not os.path.exists(settings.CALIB_IMAGES_DIR):
    print(f"❌ Calibration images directory not found: {settings.CALIB_IMAGES_DIR}")
    print("Please take calibration photos first.")
    exit(1)

# === SAVE LETTERBOX PREVIEW IMAGES ===
utils.save_letterboxed_images_preview(settings.CALIB_IMAGES_DIR, num_preview=3)

# === COLLECT CORNERS FROM ALL CALIBRATION IMAGES ===
all_corners = []
all_ids = []
image_size = settings.YOLO_INPUT_SHAPE  # Fixed size for letterboxed images
valid_images = []
skipped_images = []
letterbox_stats = []

images = glob.glob(os.path.join(settings.CALIB_IMAGES_DIR, "*.jpg"))
images.extend(glob.glob(os.path.join(settings.CALIB_IMAGES_DIR, "*.png")))
print(f"\nFound {len(images)} calibration images.")

if len(images) == 0:
    print(f"❌ No images found in {settings.CALIB_IMAGES_DIR}")
    exit(1)

for img_path in images:
    print(f"Processing: {os.path.basename(img_path)}", end=" ")
    
    # Load original image
    image = cv2.imread(img_path)
    if image is None:
        print("❌ Failed to load")
        skipped_images.append((img_path, "Failed to load image"))
        continue
    
    original_shape = image.shape[:2]
    
    # Apply YOLO letterboxing
    letterboxed_image, scale_ratio, padding = utils.letterbox_yolo_style(
        image, (utils.YOLO_INPUT_SIZE, utils.YOLO_INPUT_SIZE), utils.LETTERBOX_COLOR
    )
    
    # Store letterbox statistics
    letterbox_stats.append({
        'image': os.path.basename(img_path),
        'original_shape': original_shape,
        'scale_ratio': scale_ratio,
        'padding': padding
    })
    
    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(letterboxed_image, cv2.COLOR_BGR2GRAY)
    
    # Verify final size
    if gray.shape != (utils.YOLO_INPUT_SIZE, utils.YOLO_INPUT_SIZE):
        print(f"❌ Letterbox failed: got {gray.shape}, expected ({utils.YOLO_INPUT_SIZE}, {utils.YOLO_INPUT_SIZE})")
        skipped_images.append((img_path, f"Letterbox size mismatch: {gray.shape}"))
        continue

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is not None and len(ids) >= settings.MIN_MARKERS_PER_IMAGE:
        # Interpolate ChArUco corners
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=charuco_board
        )

        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= settings.MIN_CORNERS_PER_IMAGE:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            valid_images.append(img_path)
            print(f"✅ {len(ids)} markers, {len(charuco_ids)} corners (scale: {scale_ratio:.3f})")
        else:
            corners_found = len(charuco_ids) if charuco_ids is not None else 0
            print(f"❌ Not enough corners: {corners_found}")
            skipped_images.append((img_path, f"Only {corners_found} ChArUco corners found"))
    else:
        markers_found = len(ids) if ids is not None else 0
        print(f"❌ Not enough markers: {markers_found}")
        skipped_images.append((img_path, f"Only {markers_found} markers found"))

print("\n" + "=" * 50)
print(f"Valid images: {len(valid_images)}")
print(f"Skipped images: {len(skipped_images)}")

# Print letterbox statistics
if letterbox_stats:
    print(f"\nLetterbox Statistics:")
    scale_ratios = [s['scale_ratio'] for s in letterbox_stats]
    print(f"  Scale ratio range: {min(scale_ratios):.4f} - {max(scale_ratios):.4f}")
    print(f"  Average scale ratio: {np.mean(scale_ratios):.4f}")
    
    # Show padding statistics
    paddings_w = [s['padding'][0] for s in letterbox_stats]
    paddings_h = [s['padding'][1] for s in letterbox_stats]
    print(f"  Width padding range: {min(paddings_w):.1f} - {max(paddings_w):.1f} pixels")
    print(f"  Height padding range: {min(paddings_h):.1f} - {max(paddings_h):.1f} pixels")

if skipped_images:
    print("\nSkipped images:")
    for img_path, reason in skipped_images:
        print(f"  {os.path.basename(img_path)}: {reason}")

# === CALIBRATE IF ENOUGH VALID IMAGES FOUND ===
if len(all_corners) < settings.MIN_VALID_IMAGES:
    print(f"\n❌ Not enough valid calibration images. Found {len(all_corners)}, need at least {settings.MIN_VALID_IMAGES}.")
    print("\nTips for better calibration images:")
    print("- Ensure the entire ChArUco board is visible after letterboxing")
    print("- Take photos from different angles and distances")
    print("- Ensure good lighting with no shadows on the board")
    print("- Keep the board flat and fully in focus")
    print("- Consider that letterboxing may crop edges - keep board more centered")
    exit(1)

print(f"\n✅ Proceeding with calibration using {len(all_corners)} letterboxed images...")

# Perform calibration on letterboxed images
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=charuco_board,
    imageSize=image_size,  # (640, 640)
    cameraMatrix=None,
    distCoeffs=None
)

# === EVALUATE CALIBRATION QUALITY ===
print("\n" + "=" * 50)
print("LETTERBOXED CALIBRATION RESULTS")
print("=" * 50)

print(f"Calibration RMS error: {ret:.4f} pixels")
print("Camera Matrix (for 640x640 letterboxed images):")
print(camera_matrix)
print("Distortion Coefficients:")
print(dist_coeffs)

# Calculate expected focal length range
average_scale = np.mean([s['scale_ratio'] for s in letterbox_stats])
expected_focal_length = 3104 * average_scale  # Your original fx was ~3104
print(f"\nSanity check:")
print(f"  Average scale ratio: {average_scale:.4f}")
print(f"  Expected focal length: {expected_focal_length:.1f} pixels")
print(f"  Actual focal length: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")

# Check if focal lengths are reasonable
fx, fy = camera_matrix[0,0], camera_matrix[1,1]
if 400 < fx < 800 and 400 < fy < 800:
    print("  ✅ Focal lengths look reasonable for 640x640 images")
else:
    print("  ⚠️  Focal lengths seem unusual - check your calibration")

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
os.makedirs(os.path.dirname(settings.CALIB_OUTPUT_FILE), exist_ok=True)

# Calculate average letterbox parameters for metadata
avg_scale = np.mean([s['scale_ratio'] for s in letterbox_stats])
avg_padding_w = np.mean([s['padding'][0] for s in letterbox_stats])
avg_padding_h = np.mean([s['padding'][1] for s in letterbox_stats])

calib_data = {
    'camera_matrix': camera_matrix.tolist(),
    'dist_coeff': dist_coeffs.tolist(),
    'image_width': settings.YOLO_INPUT_SIZE,
    'image_height': settings.YOLO_INPUT_SIZE,
    'square_length_m': settings.SQUARE_LENGTH,
    'marker_length_m': settings.MARKER_LENGTH,
    'charuco_dict': settings.ARUCO_DICT,
    'calibration_rms_error': float(ret),
    'mean_reprojection_error': float(mean_error),
    'num_calibration_images': len(all_corners),
    'calibration_quality': quality,
    # YOLO-specific metadata
    'yolo_letterbox': True,
    'yolo_input_size': utils.YOLO_INPUT_SIZE,
    'letterbox_color': utils.LETTERBOX_COLOR,
    'average_scale_ratio': float(avg_scale),
    'average_padding_w': float(avg_padding_w),
    'average_padding_h': float(avg_padding_h),
    'calibration_type': 'letterboxed_for_yolo',
    'coordinate_system': '640x640_letterboxed_space'
}

with open(settings.CALIB_OUTPUT_FILE, "w") as f:
    yaml.dump(calib_data, f, default_flow_style=False)

print(f"\n📁 Saved letterboxed calibration to: {settings.CALIB_OUTPUT_FILE}")
print("\n✅ YOLO-compatible calibration complete!")

# === VERIFICATION RECOMMENDATION ===
print("\n" + "=" * 50)
print("NEXT STEPS")
print("=" * 50)
print("1. ✅ Your calibration is now compatible with YOLO's 640x640 letterboxed input")
print("2. 🔍 Check the letterbox preview images in letterbox_preview/ folder")
print("3. 📊 Compare distance measurements with known objects")
print(f"4. 🚀 Use this calibration file in your YOLO distance measurement: {settings.CALIB_OUTPUT_FILE}")
print("\nIMPORTANT NOTES:")
print(f"- This calibration is ONLY valid for 640x640 letterboxed images")
print(f"- Your YOLO predictions are already in the correct coordinate space")
print(f"- No coordinate conversion needed in your distance calculation script")
print(f"- Average scale factor was {avg_scale:.4f} (original → letterboxed)")