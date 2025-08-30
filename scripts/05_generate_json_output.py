#   05_generate_json_output.py - UPDATED with Reference Plane Fixes
from ultralytics import YOLO
import cv2
import numpy as np
import yaml
import json
import os
from pathlib import Path
from datetime import datetime
import scripts.utils as utils


def generate_json_output():
    """Main function to process S20+ images and generate JSON output"""

    # === CONFIGURATION ===
    model_path = "yolo_training_output/yolo11s-seg_frame_detector/weights/best.pt"
    s20plus_images_dir = "s20plus_images_with_ChArUco"
    calib_file = (
        "config/s20plus_calib_640x640_letterbox.yaml"  # Use letterbox calibration
    )
    output_file = "detection_results_segmentation_fixed.json"  # New filename
    confidence_threshold = 0.75

    # === LOAD MODEL AND CALIBRATION ===
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using 03_train_yolo.py")
        return

    if not os.path.exists(calib_file):
        print(f"❌ Calibration file not found: {calib_file}")
        print("Please run camera calibration first.")
        return

    if not os.path.exists(s20plus_images_dir):
        print(f"❌ S20+ images directory not found: {s20plus_images_dir}")
        return

    print("Loading YOLO segmentation model...")
    model = YOLO(model_path)

    print("Loading camera calibration...")
    calib_data = utils.load_camera_calibration(calib_file)
    print(
        f"Calibration loaded: {calib_data['image_width']}×{calib_data['image_height']} letterboxed images"
    )

    # === GET S20+ IMAGES ===
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    s20plus_images = []
    for ext in image_extensions:
        s20plus_images.extend(list(Path(s20plus_images_dir).glob(ext)))

    if not s20plus_images:
        print(f"❌ No images found in: {s20plus_images_dir}")
        return

    print(f"Found {len(s20plus_images)} S20+ images to process")

    # === PROCESS ALL IMAGES ===
    all_results = []
    stats = {
        "total_images": len(s20plus_images),
        "images_with_charuco": 0,
        "images_with_detections": 0,
        "total_detections": 0,
        "measurements_calculated": 0,
        "mask_measurements_calculated": 0,
        "bbox_measurements_calculated": 0,
        "failed_pose_estimates": 0,
    }

    for i, img_path in enumerate(s20plus_images):
        print(f"Processing {i+1}/{len(s20plus_images)}: {img_path.name}")

        result = utils.process_s20plus_image(
            img_path, model, calib_data, confidence_threshold
        )

        if result:
            all_results.append(result)

            # Update statistics
            if result["charuco_board_detected"]:
                stats["images_with_charuco"] += 1

            if result["total_detections"] > 0:
                stats["images_with_detections"] += 1
                stats["total_detections"] += result["total_detections"]

                # Count measurements
                for detection in result["detections"]:
                    if detection["real_world_measurements"] is not None:
                        stats["measurements_calculated"] += 1

                        # Check if measurements came from mask or bbox
                        if "polygon_points_3d" in detection["real_world_measurements"]:
                            stats["mask_measurements_calculated"] += 1
                        else:
                            stats["bbox_measurements_calculated"] += 1
                    elif detection["charuco_detected"]:
                        stats["failed_pose_estimates"] += 1

            print(
                f"  Detections: {result['total_detections']}, ChArUco: {'Yes' if result['charuco_board_detected'] else 'No'}"
            )

    # === SAVE RESULTS ===
    output_data = {
        "metadata": {
            "generation_timestamp": datetime.now().isoformat(),
            "model_used": model_path,
            "model_type": "YOLOv11 Segmentation",
            "calibration_file": calib_file,
            "confidence_threshold": confidence_threshold,
            "total_images_processed": len(all_results),
            "improvements": [
                "Consistent letterboxing applied",
                "RANSAC pose estimation",
                "Stricter ChArUco detection requirements",
                "Distance filtering for pose validation",
                "Measurements in centimeters",
            ],
        },
        "statistics": stats,
        "results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    # === SUMMARY ===
    print("\n" + "=" * 50)
    print("JSON GENERATION SUMMARY (IMPROVED)")
    print("=" * 50)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with ChArUco board: {stats['images_with_charuco']}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Real-world measurements calculated: {stats['measurements_calculated']}")
    print(f"  - From segmentation masks: {stats['mask_measurements_calculated']}")
    print(f"  - From bounding obb: {stats['bbox_measurements_calculated']}")
    print(f"Failed pose estimates (filtered out): {stats['failed_pose_estimates']}")
    print(f"JSON output saved to: {output_file}")

    return output_file


if __name__ == "__main__":
    try:
        output_file = generate_json_output()
        print(f"\n✅ JSON output generated successfully: {output_file}")

        # Optionally load and display a sample
        with open(output_file, "r") as f:
            data = json.load(f)

        print("\nSample detection (first result with measurements):")
        for result in data["results"]:
            for detection in result["detections"]:
                if detection["real_world_measurements"]:
                    measurements = detection["real_world_measurements"]
                    if "polygon_points_3d" in measurements:
                        print(
                            f"  {detection['class_name']} (from mask): {measurements['area_cm2']:.2f}cm² (width: {measurements['width_cm']:.1f}cm, height: {measurements['height_cm']:.1f}cm)"
                        )
                    else:
                        print(
                            f"  {detection['class_name']} (from bbox): {measurements['width_cm']:.1f}cm × {measurements['height_cm']:.1f}cm"
                        )
                    break
            else:
                continue
            break

    except Exception as e:
        print(f"❌ Error generating JSON output: {e}")
        raise
