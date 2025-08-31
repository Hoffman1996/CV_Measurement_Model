import cv2
import numpy as np
import os
import glob
import scripts.utils as utils
import config.settings as settings


print("=== ChArUco Camera Calibration for YOLO (640x640 Letterboxed) ===")
print(f"Board: {settings.CHARUCOBOARD_COLCOUNT}x{settings.CHARUCOBOARD_ROWCOUNT}")
print(f"Square size: {settings.SQUARE_LENGTH*1000:.1f}mm")
print(f"Marker size: {settings.MARKER_LENGTH*1000:.1f}mm")
print(f"Dictionary: {settings.ARUCO_DICT_ID}")
print(f"YOLO input size: {settings.YOLO_INPUT_SHAPE[0]}x{settings.YOLO_INPUT_SHAPE[1]}")
print(f"Letterbox padding color: {settings.LETTERBOX_COLOR}")
print("=" * 50)

# === VERIFY IMAGES DIRECTORY ===
if not os.path.exists(settings.CALIB_IMAGES_DIR):
    print(f"❌ Calibration images directory not found: {settings.CALIB_IMAGES_DIR}")
    print("Please take calibration photos first.")
    exit(1)

# Create visualization directory
visualize_corners_dir = settings.VISUALIZATION_DIR
os.makedirs(visualize_corners_dir, exist_ok=True)

# === SAVE LETTERBOX PREVIEW IMAGES ===
utils.save_letterboxed_images_preview(settings.CALIB_IMAGES_DIR, num_preview=3)

# === COLLECT CORNERS FROM ALL CALIBRATION IMAGES ===
all_corners = []
all_ids = []
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
        skipped_images.append((img_path, "❌ Failed to load image"))
        continue

    original_shape = image.shape[:2]

    # Apply YOLO letterboxing
    letterboxed_image, scale_ratio, padding = utils.letterbox_yolo_style(
        image, settings.YOLO_INPUT_SHAPE, settings.LETTERBOX_COLOR
    )

    # Store letterbox statistics
    letterbox_stats.append(
        {
            "image": os.path.basename(img_path),
            "original_shape": original_shape,
            "scale_ratio": scale_ratio,
            "padding": padding,
        }
    )

    # Verify final size
    if letterboxed_image.shape[:2] != settings.YOLO_INPUT_SHAPE:
        print(
            f"❌ Letterbox failed: got {letterboxed_image.shape}, expected {settings.YOLO_INPUT_SHAPE}"
        )
        skipped_images.append(
            (img_path, f"Letterbox size mismatch: {letterboxed_image.shape}")
        )
        continue

    markers_ids, charuco_corners, charuco_ids = utils.detect_charuco_board(
        letterboxed_image,
        settings.ARUCO_DICTt,
        utils.create_charuco_board(),
        utils.get_detector_params(),
        visualize_corners_dir,
        img_path,
    )

    all_corners.append(charuco_corners)
    all_ids.append(charuco_ids)
    valid_images.append(img_path)
    print(
        f"✅ {len(markers_ids)} markers, {len(charuco_ids)} corners (scale: {scale_ratio:.3f})"
    )

    corners_found = len(charuco_ids) if charuco_ids is not None else 0
    markers_found = len(markers_ids) if markers_ids is not None else 0
    if corners_found < settings.MIN_CORNERS_PER_IMAGE:
        print(f"❌ Not enough corners: {corners_found}")
        skipped_images.append((img_path, f"Only {corners_found} ChArUco corners found"))
    if markers_found < settings.MIN_MARKERS_PER_IMAGE:
        print(f"❌ Not enough markers: {markers_found}")
        skipped_images.append((img_path, f"Only {markers_found} markers found"))

print("\n" + "=" * 50)
print(f"Valid images: {len(valid_images)}")
print(f"Skipped images: {len(skipped_images)}")

# Print letterbox statistics
if letterbox_stats:
    print(f"\nLetterbox Statistics:")
    scale_ratios = [s["scale_ratio"] for s in letterbox_stats]
    print(f"  Scale ratio range: {min(scale_ratios):.4f} - {max(scale_ratios):.4f}")
    print(f"  Average scale ratio: {np.mean(scale_ratios):.4f}")

    # Show padding statistics
    paddings_w = [s["padding"][0] for s in letterbox_stats]
    paddings_h = [s["padding"][1] for s in letterbox_stats]
    print(
        f"  Width padding range: {min(paddings_w):.1f} - {max(paddings_w):.1f} pixels"
    )
    print(
        f"  Height padding range: {min(paddings_h):.1f} - {max(paddings_h):.1f} pixels"
    )

if skipped_images:
    print("\nSkipped images:")
    for img_path, reason in skipped_images:
        print(f"  {os.path.basename(img_path)}: {reason}")

# === CALIBRATE IF ENOUGH VALID IMAGES FOUND ===
if len(all_corners) < settings.MIN_VALID_IMAGES:
    print(
        f"\n❌ Not enough valid calibration images. Found {len(all_corners)}, need at least {settings.MIN_VALID_IMAGES}."
    )
    print("\nTips for better calibration images:")
    print("- Ensure the entire ChArUco board is visible after letterboxing")
    print("- Take photos from different angles and distances")
    print("- Ensure good lighting with no shadows on the board")
    print("- Keep the board flat and fully in focus")
    print("- Consider that letterboxing may crop edges - keep board more centered")
    exit(1)

print("\n 😎 ChArUco detection completed!")
