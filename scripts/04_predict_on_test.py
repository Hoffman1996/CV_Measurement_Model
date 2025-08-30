#   04_predict_on_test.py
from scipy import stats
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def predict_on_test_images():
    # === CONFIGURATION ===
    model_path = "yolo_training_output/yolo11s-seg_frame_detector/weights/best.pt"
    test_images_dir = (
        "datasets/yolo_dataset/test/images"  # Assuming you have test split
    )
    test_labels_dir = "datasets/yolo_dataset/test/labels"  # For comparison if available
    output_dir = "test_predictions"
    confidence_threshold = 0.8  # Higher confidence = fewer false positives
    iou_threshold = 0.01  # Lower IoU = less merging of nearby detections

    # Class names
    class_names = ["frame"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # === LOAD MODEL ===
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please run 03_train_yolo.py first to train the model.")
        print("Or update the model_path to point to your trained model.")
        return

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # === GET TEST IMAGES ===
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory not found: {test_images_dir}")
        print("Using validation images for testing...")
        test_images_dir = "datasets/yolo_dataset/valid/images"
        test_labels_dir = "datasets/yolo_dataset/valid/labels"

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(Path(test_images_dir).glob(ext)))

    if not test_images:
        print(f"❌ No test images found in: {test_images_dir}")
        return

    print(f"Found {len(test_images)} test images")

    # === PREDICTION STATISTICS ===
    stats = {
        "total_images": len(test_images),
        "images_with_detections": 0,
        "total_detections": 0,
        "frame_detections": 0,
    }

    # === PROCESS EACH IMAGE ===
    for i, img_path in enumerate(test_images):
        print(f"Processing {i+1}/{len(test_images)}: {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Make prediction
        results = model.predict(
            task="segment",  # Changed from 'obb' to 'segment'
            source=str(img_path),
            conf=confidence_threshold,
            iou=iou_threshold,
            save=False,
            verbose=False,
            max_det=10,  # Limit maximum detections per image
            agnostic_nms=True,  # Class-agnostic NMS
            retina_masks=True,  # Higher quality masks
        )

        # Process results
        result = results[0]  # Get first (and only) result

        if len(result.obb) > 0:  # Changed from result.obb to result.obb
            stats["images_with_detections"] += 1
            stats["total_detections"] += len(result.obb)

            # Create visualization with both obb and masks
            annotated_image = result.plot(conf=True, labels=True, obb=True, masks=True)

            # Count detections by class
            for box in result.obb:
                stats["frame_detections"] += 1

            # Save annotated image
            output_path = os.path.join(output_dir, f"pred_{img_path.name}")
            cv2.imwrite(output_path, annotated_image)

            # Save mask overlay separately
            mask_output_path = os.path.join(output_dir, f"mask_{img_path.name}")
            save_mask_overlay(image, result, mask_output_path)

            # Print detection details
            print(f"  Detections found: {len(result.obb)}")
            for j, box in enumerate(result.obb):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                print(
                    f"    {j+1}. {class_names[class_id]}: {confidence:.3f} at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                )

                # Print mask area if available
                if result.masks is not None and j < len(result.masks.data):
                    mask = result.masks.data[j].cpu().numpy()
                    mask_area = np.sum(mask > 0)
                    print(f"        Mask area: {mask_area} pixels")
        else:
            print(f"  No detections found")

    # === SUMMARY STATISTICS ===
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {stats['total_images']}")
    print(
        f"Images with detections: {stats['images_with_detections']} ({stats['images_with_detections']/stats['total_images']*100:.1f}%)"
    )
    print(f"Total detections: {stats['total_detections']}")
    print(f"Frame detections: {stats['frame_detections']}")
    print(
        f"Average detections per image: {stats['total_detections']/stats['total_images']:.2f}"
    )
    print(f"Annotated images saved to: {output_dir}")
    print(f"Mask overlays saved to: {output_dir} (mask_*.jpg)")

    # === SAVE SUMMARY TO FILE ===
    summary_file = os.path.join(output_dir, "prediction_summary.txt")
    with open(summary_file, "w") as f:
        f.write("YOLO MODEL PREDICTION SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model used: {model_path}\n")
        f.write(f"Model type: YOLOv11 Segmentation\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n")
        f.write(f"IoU threshold: {iou_threshold}\n")
        f.write(f"Test images directory: {test_images_dir}\n")
        f.write(f"Total images processed: {stats['total_images']}\n")
        f.write(
            f"Images with detections: {stats['images_with_detections']} ({stats['images_with_detections']/stats['total_images']*100:.1f}%)\n"
        )
        f.write(f"Total detections: {stats['total_detections']}\n")
        f.write(f"Frame detections: {stats['frame_detections']}\n")
        f.write(
            f"Average detections per image: {stats['total_detections']/stats['total_images']:.2f}\n"
        )

    print(f"Summary saved to: {summary_file}")


def save_mask_overlay(original_image, result, output_path):
    """Save an overlay of masks on the original image"""
    if result.masks is None:
        return

    overlay = original_image.copy()

    # Create colored masks
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for i, mask_data in enumerate(result.masks.data):
        mask = mask_data.cpu().numpy()

        # Resize mask to match image dimensions if needed
        if mask.shape != original_image.shape[:2]:
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

        # Create colored mask
        color = colors[i % len(colors)]
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask > 0.5] = color

        # Blend with original image
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

    cv2.imwrite(output_path, overlay)


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

    mask_images = list(Path(output_dir).glob("mask_*.jpg"))
    mask_images.extend(list(Path(output_dir).glob("mask_*.png")))

    if not pred_images:
        print("No prediction images found to visualize.")
        return

    # Select random images
    import random

    selected_images = random.sample(pred_images, min(num_images, len(pred_images)))

    # Create subplot for predictions and masks
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, img_path in enumerate(selected_images):
        if i >= 4:  # Limit to 4 images
            break

        # Load prediction image
        pred_img = cv2.imread(str(img_path))
        pred_img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

        axes[0, i].imshow(pred_img_rgb)
        axes[0, i].set_title(f"Prediction: {img_path.name}", fontsize=10)
        axes[0, i].axis("off")

        # Load corresponding mask image
        mask_path = img_path.parent / img_path.name.replace("pred_", "mask_")
        if mask_path.exists():
            mask_img = cv2.imread(str(mask_path))
            mask_img_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

            axes[1, i].imshow(mask_img_rgb)
            axes[1, i].set_title(f"Mask Overlay: {mask_path.name}", fontsize=10)
            axes[1, i].axis("off")
        else:
            axes[1, i].axis("off")

    # Hide unused subplots
    for i in range(len(selected_images), 4):
        axes[0, i].axis("off")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run predictions on test images
    predict_on_test_images()

    # Optionally visualize some results
    print("\nWould you like to visualize some prediction results? (y/n)")
    response = input().lower().strip()
    if response in ["y", "yes"]:
        visualize_random_predictions()
