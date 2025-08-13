import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from ultralytics import YOLO

def load_camera_calibration(calib_file="config/s20plus_calib.yaml"):
    """Load camera calibration parameters"""
    with open(calib_file, 'r') as f:
        calib_data = yaml.safe_load(f)
    
    camera_matrix = np.array(calib_data['camera_matrix'])
    dist_coeffs = np.array(calib_data['dist_coeff'])
    
    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'image_width': calib_data['image_width'],
        'image_height': calib_data['image_height'],
        'square_length_m': calib_data['square_length_m'],
        'marker_length_m': calib_data['marker_length_m'],
        'charuco_dict': calib_data['charuco_dict']
    }

def detect_and_draw_charuco(image, calib_data):
    """Detect ChArUco board and draw it on the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(calib_data['charuco_dict'])
    charuco_board = cv2.aruco.CharucoBoard(
        size=(5, 5),
        squareLength=calib_data['square_length_m'],
        markerLength=calib_data['marker_length_m'],
        dictionary=aruco_dict
    )
    
    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    
    # Draw detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
        # Try to detect ChArUco corners
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=charuco_board
        )
        
        # Draw ChArUco corners
        if charuco_corners is not None and len(charuco_corners) > 4:
            cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids, (0, 255, 0))
            
            # Draw coordinate axis if we have enough corners
            if len(charuco_corners) >= 4:
                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=charuco_board.getChessboardCorners()[charuco_ids.flatten()],
                    imagePoints=charuco_corners,
                    cameraMatrix=calib_data['camera_matrix'],
                    distCoeffs=calib_data['dist_coeffs']
                )
                
                if success:
                    # Draw 3D coordinate axes
                    cv2.drawFrameAxes(image, calib_data['camera_matrix'], calib_data['dist_coeffs'], 
                                    rvec, tvec, 0.1, 3)
            
            return charuco_corners, charuco_ids
    
    return None, None

def visualize_detection_with_charuco(image_path, model_path, calib_data, confidence_threshold=0.25):
    """Visualize YOLO detections along with ChArUco board detection"""
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Make a copy for visualization
    vis_image = image.copy()
    
    # Load YOLO model and make predictions
    model = YOLO(model_path)
    results = model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)
    result = results[0]
    
    # Detect and draw ChArUco board
    charuco_corners, charuco_ids = detect_and_draw_charuco(vis_image, calib_data)
    
    # Draw YOLO detections
    detection_info = []
    for i, box in enumerate(result.obb):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        
        # Draw bounding box
        cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
        
        # Add label with confidence and measurements
        label_text = f"{model.names[class_id]}: {confidence:.2f}"
        
        # Calculate measurements if ChArUco is detected
        if charuco_corners is not None:
            # Simple measurement calculation (you can use your existing function here)
            width_px = bbox[2] - bbox[0]
            height_px = bbox[3] - bbox[1]
            
            # For display purposes, show pixel dimensions
            measurement_text = f"{width_px}px x {height_px}px"
            
            # Store detection info
            detection_info.append({
                'id': i,
                'class': model.names[class_id],
                'confidence': confidence,
                'bbox': bbox,
                'width_px': width_px,
                'height_px': height_px
            })
        else:
            measurement_text = "No ChArUco detected"
        
        # Draw label background
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        measure_size = cv2.getTextSize(measurement_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        label_y = bbox[1] - 10
        if label_y < 0:
            label_y = bbox[3] + 30
            
        # Draw label rectangles
        cv2.rectangle(vis_image, (bbox[0], label_y - label_size[1] - 25), 
                     (bbox[0] + max(label_size[0], measure_size[0]) + 10, label_y + 5), 
                     (0, 0, 255), -1)
        
        # Draw text
        cv2.putText(vis_image, label_text, (bbox[0] + 5, label_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, measurement_text, (bbox[0] + 5, label_y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add image info text
    info_text = [
        f"Image: {Path(image_path).name}",
        f"Detections: {len(result.obb)}",
        f"ChArUco corners: {len(charuco_corners) if charuco_corners is not None else 0}",
        f"Confidence threshold: {confidence_threshold}"
    ]
    
    # Draw info background
    y_offset = 30
    for i, text in enumerate(info_text):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(vis_image, (10, y_offset + i*25 - text_size[1] - 5), 
                     (10 + text_size[0] + 10, y_offset + i*25 + 5), (0, 0, 0), -1)
        cv2.putText(vis_image, text, (15, y_offset + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_image, detection_info, charuco_corners is not None

def visualize_from_json_results(json_file="detection_results.json", num_images=4, save_visualizations=True):
    """Visualize detections using results from JSON file"""
    
    # Load configuration
    model_path = "yolo_training_output/yolov8s-obb_frame_detector/weights/best.pt"
    calib_file = "config/s20plus_calib.yaml"
    output_dir = "visualization_output"
    
    if save_visualizations:
        Path(output_dir).mkdir(exist_ok=True)
    
    # Load calibration data
    calib_data = load_camera_calibration(calib_file)
    
    # Load JSON results
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Select images to visualize
    results_to_show = data['results'][:num_images]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, result in enumerate(results_to_show):
        if i >= 4:
            break
            
        image_path = result['image_path']
        
        print(f"Processing {i+1}/{min(num_images, len(results_to_show))}: {result['image_name']}")
        
        # Create visualization
        vis_image, detection_info, has_charuco = visualize_detection_with_charuco(
            image_path, model_path, calib_data
        )
        
        if vis_image is not None:
            # Convert BGR to RGB for matplotlib
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            
            # Display in subplot
            axes[i].imshow(vis_image_rgb)
            axes[i].set_title(f"{result['image_name']}\n"
                            f"Detections: {result['total_detections']} | "
                            f"ChArUco: {'Yes' if has_charuco else 'No'}", 
                            fontsize=10)
            axes[i].axis('off')
            
            # Save individual visualization
            if save_visualizations:
                save_path = Path(output_dir) / f"vis_{result['image_name']}"
                cv2.imwrite(str(save_path), vis_image)
                print(f"  Saved: {save_path}")
            
            # Print detection details
            print(f"  Detections found: {len(detection_info)}")
            for det in detection_info:
                print(f"    {det['id']}: {det['class']} ({det['confidence']:.2f}) - {det['width_px']}×{det['height_px']} px")
            
            # Print measurements from JSON
            for det in result['detections']:
                if det['real_world_measurements']:
                    w = det['real_world_measurements']['width_m']
                    h = det['real_world_measurements']['height_m']
                    print(f"    Real measurements: {w:.2f}m × {h:.2f}m")
    
    # Hide unused subplots
    for i in range(len(results_to_show), 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    if save_visualizations:
        print(f"\n📁 Visualizations saved to: {output_dir}")

def visualize_single_image(image_path, model_path=None, calib_file="config/s20plus_calib.yaml"):
    """Visualize a single image with detections and ChArUco board"""
    
    if model_path is None:
        model_path = "yolo_training_output/yolov8s-obb_frame_detector/weights/best.pt"
    
    # Load calibration data
    calib_data = load_camera_calibration(calib_file)
    
    # Create visualization
    vis_image, detection_info, has_charuco = visualize_detection_with_charuco(
        image_path, model_path, calib_data
    )
    
    if vis_image is not None:
        # Display using matplotlib
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(vis_image_rgb)
        plt.title(f"Detection Visualization: {Path(image_path).name}")
        plt.axis('off')
        plt.show()
        
        # Print details
        print(f"Image: {Path(image_path).name}")
        print(f"ChArUco board detected: {has_charuco}")
        print(f"Number of detections: {len(detection_info)}")
        
        for det in detection_info:
            print(f"  {det['class']}: {det['confidence']:.2f} confidence, {det['width_px']}×{det['height_px']} pixels")
    
    return vis_image

if __name__ == "__main__":
    print("=== YOLO Detection + ChArUco Visualization ===")
    print()
    
    # Option 1: Visualize from JSON results
    print("1. Visualizing from JSON results...")
    try:
        visualize_from_json_results("detection_results.json", num_images=4, save_visualizations=True)
    except FileNotFoundError:
        print("❌ detection_results.json not found. Run 05_generate_json_output.py first.")
    except Exception as e:
        print(f"❌ Error visualizing from JSON: {e}")
    
    print()
    
    # Option 2: Visualize a single image (example)
    print("2. Example single image visualization...")
    single_image_path = "s20plus_images_with_ChArUco/20250804_122919.jpg"  # Update this path
    
    if Path(single_image_path).exists():
        try:
            visualize_single_image(single_image_path)
        except Exception as e:
            print(f"❌ Error visualizing single image: {e}")
    else:
        print(f"❌ Image not found: {single_image_path}")
        print("Update the single_image_path variable to point to an existing image.")