from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def predict_on_test_images():
    # === CONFIGURATION ===
    # Update this path after training is complete
    model_path = "yolo_training_output/yolov8s_window_door_detector/weights/best.pt"
    test_images_dir = "datasets/yolo_dataset/test/images"  # Assuming you have test split
    test_labels_dir = "datasets/yolo_dataset/test/labels"  # For comparison if available
    output_dir = "test_predictions"
    confidence_threshold = 0.25
    iou_threshold = 0.5
    
    # Class names
    class_names = ['window', 'door']
    
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
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(Path(test_images_dir).glob(ext)))
    
    if not test_images:
        print(f"❌ No test images found in: {test_images_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # === PREDICTION STATISTICS ===
    stats = {
        'total_images': len(test_images),
        'images_with_detections': 0,
        'total_detections': 0,
        'window_detections': 0,
        'door_detections': 0
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
            task='obb', # Use 'obb' for oriented bounding boxes
            source=str(img_path),
            conf=confidence_threshold,
            iou=iou_threshold,
            save=False,
            verbose=False
        )
        
        # Process results
        result = results[0]  # Get first (and only) result
        
        if len(result.boxes) > 0:
            stats['images_with_detections'] += 1
            stats['total_detections'] += len(result.boxes)
            
            # Create visualization
            annotated_image = result.plot(conf=True, labels=True, boxes=True)
            
            # Count detections by class
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id == 0:  # window
                    stats['window_detections'] += 1
                elif class_id == 1:  # door
                    stats['door_detections'] += 1
            
            # Save annotated image
            output_path = os.path.join(output_dir, f"pred_{img_path.name}")
            cv2.imwrite(output_path, annotated_image)
            
            # Print detection details
            print(f"  Detections found: {len(result.boxes)}")
            for j, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                print(f"    {j+1}. {class_names[class_id]}: {confidence:.3f} at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
        else:
            print(f"  No detections found")
    
    # === SUMMARY STATISTICS ===
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with detections: {stats['images_with_detections']} ({stats['images_with_detections']/stats['total_images']*100:.1f}%)")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Window detections: {stats['window_detections']}")
    print(f"Door detections: {stats['door_detections']}")
    print(f"Average detections per image: {stats['total_detections']/stats['total_images']:.2f}")
    print(f"Annotated images saved to: {output_dir}")
    
    # === SAVE SUMMARY TO FILE ===
    summary_file = os.path.join(output_dir, "prediction_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("YOLO MODEL PREDICTION SUMMARY\n")
        f.write("="*40 + "\n")
        f.write(f"Model used: {model_path}\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n")
        f.write(f"IoU threshold: {iou_threshold}\n")
        f.write(f"Test images directory: {test_images_dir}\n")
        f.write(f"Total images processed: {stats['total_images']}\n")
        f.write(f"Images with detections: {stats['images_with_detections']} ({stats['images_with_detections']/stats['total_images']*100:.1f}%)\n")
        f.write(f"Total detections: {stats['total_detections']}\n")
        f.write(f"Window detections: {stats['window_detections']}\n")
        f.write(f"Door detections: {stats['door_detections']}\n")
        f.write(f"Average detections per image: {stats['total_detections']/stats['total_images']:.2f}\n")
    
    print(f"Summary saved to: {summary_file}")

def visualize_random_predictions(num_images=4):
    """Display a few random predictions for quick visual inspection"""
    output_dir = "test_predictions"
    
    if not os.path.exists(output_dir):
        print(f"No predictions found in {output_dir}. Run predict_on_test_images() first.")
        return
    
    pred_images = list(Path(output_dir).glob("pred_*.jpg"))
    pred_images.extend(list(Path(output_dir).glob("pred_*.png")))
    
    if not pred_images:
        print("No prediction images found to visualize.")
        return
    
    # Select random images
    import random
    selected_images = random.sample(pred_images, min(num_images, len(pred_images)))
    
    # Create subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, img_path in enumerate(selected_images):
        if i >= 4:  # Limit to 4 images
            break
            
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Prediction: {img_path.name}", fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(selected_images), 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run predictions on test images
    predict_on_test_images()
    
    # Optionally visualize some results
    print("\nWould you like to visualize some prediction results? (y/n)")
    response = input().lower().strip()
    if response in ['y', 'yes']:
        visualize_random_predictions()