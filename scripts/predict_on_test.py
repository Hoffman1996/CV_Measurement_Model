import os
import cv2
import time
import random
import numpy as np
from scipy import stats
from pathlib import Path
import scripts.utils as utils
import matplotlib.pyplot as plt
import config.settings as settings

test_dir = settings.unlabeled_images_dir
# test_dir = settings.TEST_IMAGES_DIR


def get_test_images():
    """Get all test images from the test directory."""
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    test_images = []

    for ext in image_extensions:
        test_images.extend(list(Path(test_dir).glob(ext)))

    if not test_images:
        raise FileNotFoundError(f"No test images found in: {test_dir}")

    print(f"Found {len(test_images)} test images")
    return test_images


def initialize_stats(total_images):
    """Initialize statistics dictionary for tracking results."""
    return {
        "total_images": total_images,
        "images_with_detections": 0,
        "total_detections": 0,
        "confidence_scores": [],
        "inference_times": [],
    }


def process_detection_results(result, img_path, output_dir, class_names):
    """Process detection results and save visualizations."""
    detections_info = []

    if len(result.obb) > 0:
        # Create visualization
        annotated_image = result.plot(conf=True, labels=True)

        # Process each detection
        for j, box in enumerate(result.obb):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            obb_coords = box.xyxyxyxy[0].cpu().numpy()

            detection_info = {
                "class_name": class_names[class_id],
                "confidence": confidence,
                "coordinates": obb_coords,
            }
            detections_info.append(detection_info)

            print(f"    {j+1}. {class_names[class_id]}: {confidence:.3f}")
            print(f"        OBB corners: {obb_coords}")

        # Save annotated image
        output_path = os.path.join(output_dir, f"pred_{img_path.name}")
        cv2.imwrite(output_path, annotated_image)

    return detections_info


def update_statistics(stats, detections_info, inference_time):
    """Update statistics with results from current image."""
    if detections_info:
        stats["images_with_detections"] += 1
        stats["total_detections"] += len(detections_info)

        for detection in detections_info:
            stats["confidence_scores"].append(detection["confidence"])

    stats["inference_times"].append(inference_time)


def print_summary_statistics(stats):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {stats['total_images']}")
    print(
        f"Images with detections: {stats['images_with_detections']} "
        f"({stats['images_with_detections']/stats['total_images']*100:.1f}%)"
    )
    print(f"Total detections: {stats['total_detections']}")
    print(
        f"Average detections per image: {stats['total_detections']/stats['total_images']:.2f}"
    )

    if stats["inference_times"]:
        print(f"Average inference time: {np.mean(stats['inference_times']):.3f}s")
        print(
            f"Inference time range: {min(stats['inference_times']):.3f}s - "
            f"{max(stats['inference_times']):.3f}s"
        )

    if stats["confidence_scores"]:
        print(
            f"Confidence scores - Min: {min(stats['confidence_scores']):.3f}, "
            f"Max: {max(stats['confidence_scores']):.3f}, "
            f"Mean: {np.mean(stats['confidence_scores']):.3f}"
        )


def save_summary_to_file(
    stats, best_model_path, confidence_threshold, iou_threshold, output_dir
):
    """Save detailed summary to text file."""
    summary_file = os.path.join(output_dir, "prediction_summary.txt")

    with open(summary_file, "w") as f:
        f.write("YOLO OBB MODEL PREDICTION SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model used: {best_model_path}\n")
        f.write(f"Model type: {settings.MODEL_ARCHITECTURE}\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n")
        f.write(f"IoU threshold: {iou_threshold}\n")
        f.write(f"Test images directory: {test_dir}\n")
        f.write(f"Total images processed: {stats['total_images']}\n")
        f.write(
            f"Images with detections: {stats['images_with_detections']} "
            f"({stats['images_with_detections']/stats['total_images']*100:.1f}%)\n"
        )
        f.write(f"Total detections: {stats['total_detections']}\n")
        f.write(
            f"Average detections per image: {stats['total_detections']/stats['total_images']:.2f}\n"
        )

        if stats["inference_times"]:
            f.write(
                f"Average inference time: {np.mean(stats['inference_times']):.3f}s\n"
            )
        if stats["confidence_scores"]:
            f.write(
                f"Mean confidence score: {np.mean(stats['confidence_scores']):.3f}\n"
            )

    print(f"Summary saved to: {summary_file}")


def predict_on_test_images():
    """Main function to run predictions on test images."""
    # Configuration
    confidence_threshold = settings.CONFIDENCE_THRESHOLD
    iou_threshold = settings.IOU_THRESHOLD
    output_dir = settings.TEST_PREDICTIONS_OUTPUT_DIR
    class_names = ["frame"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load model and validate paths
        model, best_model_path = utils.load_model_and_validate_paths(test_dir)

        # Get test images
        test_images = get_test_images()

        # Initialize statistics
        stats = initialize_stats(len(test_images))

        # Process each image
        for i, img_path in enumerate(test_images):
            print(f"Processing {i+1}/{len(test_images)}: {img_path.name}")

            # Process single image
            result, inference_time, image = utils.process_single_image(
                model, img_path, confidence_threshold, iou_threshold
            )

            if result is None:
                continue

            # Process detection results
            detections_info = process_detection_results(
                result, img_path, output_dir, class_names
            )

            # Update statistics
            update_statistics(stats, detections_info, inference_time)

            # Print results for this image
            if detections_info:
                print(f"  Detections found: {len(detections_info)}")
            else:
                print(f"  No detections found")

        # Print and save summary
        print_summary_statistics(stats)
        print(f"Annotated images saved to: {output_dir}")

        save_summary_to_file(
            stats, best_model_path, confidence_threshold, iou_threshold, output_dir
        )

        return stats

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


def visualize_random_predictions(num_images=4):
    """Display a few random predictions for quick visual inspection"""
    output_dir = "test_predictions"

    if not os.path.exists(output_dir):
        print(
            f"No predictions found in {output_dir}. Run predict_on_test_images() first."
        )
        return

    pred_images = list(Path(output_dir).glob("pred_*.jpg"))
    pred_images.extend(list(Path(output_dir).glob("pred_*.png")))

    if not pred_images:
        print("No prediction images found to visualize.")
        return

    # Select random images
    selected_images = random.sample(pred_images, min(num_images, len(pred_images)))

    # Create subplot for predictions and masks
    cols = min(4, len(selected_images))
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

    if cols == 1:
        axes = [axes]

    for i, img_path in enumerate(selected_images):
        # Load prediction image
        pred_img = cv2.imread(str(img_path))
        pred_img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

        axes[i].imshow(pred_img_rgb)
        axes[i].set_title(f"Prediction: {img_path.name}", fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        stats = predict_on_test_images()
        print("Testing completed successfully!")

        # Optional visualization (non-interactive)
        print("Displaying random prediction samples...")
        visualize_random_predictions()

    except Exception as e:
        print(f"Testing failed: {e}")
        raise
