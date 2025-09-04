import os
import yaml
import torch
from pathlib import Path
import scripts.utils as utils
from ultralytics import YOLO
import config.settings as settings


def train_yolo_model():
    # === CONFIGURATION ===
    model_arch = settings.MODEL_ARCHITECTURE
    data_yaml = settings.DATA_YAML_FOR_OBB
    imgsz = settings.YOLO_INPUT_SIZE
    epochs = settings.TRAINING_EPOCHS
    batch = settings.TRAINING_BATCH_SIZE
    project = settings.TRAINING_OUTPUT_DIR
    name = settings.MODEL_NAME

    # Create output directory
    os.makedirs(project, exist_ok=True)

    # Verify data.yaml exists and is properly formatted
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data config file not found: {data_yaml}")

    # Load and verify data.yaml
    with open(data_yaml, "r") as f:
        data_config = yaml.safe_load(f)

    print("=== TRAINING CONFIGURATION ===")
    print(f"Model Architecture: {model_arch}")
    print(f"Dataset Config: {data_yaml}")
    print(f"Classes: {data_config.get('names', 'Not found')}")
    print(f"Number of Classes: {data_config.get('nc', 'Not found')}")
    print(f"Image Size: {imgsz}x{imgsz}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch}")
    print(f"Output Directory: {project}/{name}")
    print("=" * 40)

    # Verify training and validation directories exist
    train_images = Path(data_config["train"])
    val_images = Path(data_config["val"])

    if not train_images.exists():
        raise FileNotFoundError(f"Training images directory not found: {train_images}")
    if not val_images.exists():
        raise FileNotFoundError(f"Validation images directory not found: {val_images}")

    # Count images in train and validation sets
    train_count = utils.count_images(train_images)
    val_count = utils.count_images(val_images)

    # Check for test set
    test_images = Path(data_config["test"])
    test_count = 0
    if test_images.exists():
        test_count = len(list(test_images.glob("*.jpg"))) + len(
            list(test_images.glob("*.png"))
        )
        print(f"Test images found: {test_count}")
    else:
        print("⚠️  No test set found. Consider creating one for final model evaluation.")

    total_images = train_count + val_count + test_count

    print(f"Training images found: {train_count}")
    print(f"Validation images found: {val_count}")
    print(f"Total dataset size: {total_images}")

    # Verify the split ratios match intended 75/15/10 split
    if total_images > 0:
        train_ratio = (train_count / total_images) * 100
        val_ratio = (val_count / total_images) * 100
        test_ratio = (test_count / total_images) * 100
        print(
            f"Dataset split: Train {train_ratio:.1f}% | Val {val_ratio:.1f}% | Test {test_ratio:.1f}%"
        )
    else:
        train_ratio = val_ratio = test_ratio = 0

    # Warn if ratios seem off from expected 75/15/10
    if abs(train_ratio - 75) > 5:
        print(f"⚠️  Training set is {train_ratio:.1f}% (expected ~75%)")
    if abs(val_ratio - 15) > 5:
        print(f"⚠️  Validation set is {val_ratio:.1f}% (expected ~15%)")
    if test_count > 0 and abs(test_ratio - 10) > 5:
        print(f"⚠️  Test set is {test_ratio:.1f}% (expected ~10%)")

    print("=" * 40)

    # === INITIALIZE AND TRAIN MODEL ===
    print("Initializing YOLO model...")
    model = YOLO(model_arch)

    yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {yolo_device}")

    print("Starting training...")
    results = model.train(
        task="segment",  # Changed from 'obb' to 'segment'
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        save=True,
        plots=True,
        val=True,
        patience=12,  # Reduced patience for smaller dataset
        device=yolo_device,
        optimizer="AdamW",
        lr0=0.0005,  # Conservative learning rate for small dataset
        lrf=0.05,
        warmup_epochs=3,
        cos_lr=True,
        # Segmentation-optimized augmentation parameters
        hsv_h=0.015,  # Slightly more hue variation
        hsv_s=0.5,  # More saturation variation
        hsv_v=0.4,  # More brightness variation
        degrees=10.0,  # More rotation for single class
        translate=0.1,  # More translation
        scale=0.3,  # More scale variation
        shear=2.0,  # Add shear augmentation
        perspective=0.0001,  # Slight perspective changes
        flipud=0.0,  # No vertical flip for frames
        fliplr=0.5,  # Keep horizontal flip
        mosaic=0.5,  # Higher mosaic for small dataset
        mixup=0.1,  # Add mixup augmentation
        copy_paste=0.1,  # Beneficial for segmentation
        erasing=0.05,  # Random erasing
        crop_fraction=1.0,  # Keep full crops for frame detection
        # Segmentation specific parameters
        overlap_mask=True,  # Allow overlapping masks
        mask_ratio=4,  # Downsample ratio for masks
        close_mosaic=15,  # Close mosaic early to stabilize training
    )

    # === TRAINING RESULTS ===
    print("\n=== TRAINING COMPLETED ===")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    print(f"Last model saved to: {results.save_dir}/weights/last.pt")

    # === VALIDATION METRICS ===
    print("\n=== VALIDATION RESULTS ===")
    try:
        # Load the best model for validation
        best_model = YOLO(f"{results.save_dir}/weights/best.pt")
        metrics = best_model.val()

        # OBB metrics
        print(f"mAP50: {metrics.seg.map50:.4f}")
        print(f"mAP50-95: {metrics.seg.map:.4f}")
        print(f"Precision: {metrics.seg.mp:.4f}")
        print(f"Recall: {metrics.seg.mr:.4f}")

        # Class-specific metrics
        if (
            hasattr(metrics.seg, "ap_class_index")
            and metrics.seg.ap_class_index is not None
        ):
            for i, class_name in enumerate(data_config["names"]):
                if i < len(metrics.seg.ap50):
                    print(f"{class_name} - AP50: {metrics.seg.ap50[i]:.4f}")

    except Exception as e:
        print(f"Error getting detailed metrics: {e}")

    # === SAVE TRAINING SUMMARY ===
    summary_file = f"{results.save_dir}/training_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=== YOLO TRAINING SUMMARY ===\n")
        f.write(f"Model Architecture: {model_arch}\n")
        f.write(f"Dataset: {data_yaml}\n")
        f.write(f"Classes: {data_config['names']}\n")
        f.write(f"Training Images: {train_count}\n")
        f.write(f"Validation Images: {val_count}\n")
        f.write(f"Test Images: {test_count}\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(
            f"Dataset Split: Train {train_ratio:.1f}% | Val {val_ratio:.1f}% | Test {test_ratio:.1f}%\n"
        )
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch}\n")
        f.write(f"Image Size: {imgsz}\n")
        f.write(f"Best Model: {results.save_dir}/weights/best.pt\n")
        f.write(f"Training Results Directory: {results.save_dir}\n")

    print(f"\nTraining summary saved to: {summary_file}")
    print("\n🎉 Training completed successfully!")
    print(f"📁 Check results in: {results.save_dir}")

    return results.save_dir


if __name__ == "__main__":
    try:
        output_dir = train_yolo_model()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
