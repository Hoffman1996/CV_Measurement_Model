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

def draw_segmentation_mask(image, mask, color, alpha=0.3):
    """Draw segmentation mask overlay on image"""
    # Resize mask to match image dimensions if needed
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0.5] = color
    
    # Blend with original image
    return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

def visualize_detection_with_charuco(image_path, model_path, calib_data, confidence_threshold=0.25):
    """Visualize YOLO segmentation detections along with ChArUco board detection"""
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Make a copy for visualization
    vis_image = image.copy()
    mask_overlay = image.copy()
    
    # Load YOLO model and make predictions
    model = YOLO(model_path)
    results = model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)
    result = results[0]
    
    # Detect and draw ChArUco board
    charuco_corners, charuco_ids = detect_and_draw_charuco(vis_image, calib_data)
    
    # Colors for different detections
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    # Draw YOLO detections
    detection_info = []
    for i, box in enumerate(result.boxes):  # Changed from result.obb to result.boxes
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy().astype(int)
        
        # Choose color for this detection
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Draw segmentation mask if available
        mask_area_px = 0
        if result.masks is not None and i < len(result.masks.data):
            mask = result.masks.data[i].cpu().numpy()
            
            # Draw mask overlay
            mask_overlay = draw_segmentation_mask(mask_overlay, mask, color, alpha=0.4)
            
            # Calculate mask area
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask_area_px = np.sum(mask > 0.5)
            
            # Draw mask contour on main image
            contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(vis_image, contours, -1, color, 2)
        
        # Add label with confidence and measurements
        label_text = f"{model.names[class_id]}: {confidence:.2f}"
        
        # Calculate measurements if ChArUco is detected
        measurement_lines = []
        if charuco_corners is not None:
            # Bbox measurements
            width_px = bbox[2] - bbox[0]
            height_px = bbox[3] - bbox[1]
            bbox_area_px = width_px * height_px
            
            measurement_lines.append(f"BBox: {width_px}×{height_px}px ({bbox_area_px:,}px²)")
            
            if mask_area_px > 0:
                measurement_lines.append(f"Mask: {mask_area_px:,}px²")
                coverage = (mask_area_px / bbox_area_px) * 100
                measurement_lines.append(f"Coverage: {coverage:.1f}%")
            
            # Store detection info
            detection_info.append({
                'id': i,
                'class': model.names[class_id],
                'confidence': confidence,
                'bbox': bbox,
                'width_px': width_px,
                'height_px': height_px,
                'bbox_area_px': bbox_area_px,
                'mask_area_px': mask_area_px,
                'color': color
            })
        else:
            measurement_lines = ["No ChArUco detected"]
        
        # Calculate label dimensions
        label_height = 20
        total_height = label_height * (len(measurement_lines) + 1) + 10
        max_width = 0
        
        # Find maximum width needed
        for line in [label_text] + measurement_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            max_width = max(max_width, text_size[0])
        
        # Position label
        label_x = bbox[0]
        label_y = bbox[1] - total_height - 5
        if label_y < 0:
            label_y = bbox[3] + 5
        
        # Draw label background
        cv2.rectangle(vis_image, 
                     (label_x, label_y), 
                     (label_x + max_width + 10, label_y + total_height), 
                     color, -1)
        
        # Draw label border
        cv2.rectangle(vis_image, 
                     (label_x, label_y), 
                     (label_x + max_width + 10, label_y + total_height), 
                     (255, 255, 255), 2)
        
        # Draw text lines
        y_offset = label_y + 15
        cv2.putText(vis_image, label_text, (label_x + 5, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for line in measurement_lines:
            y_offset += label_height
            cv2.putText(vis_image, line, (label_x + 5, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Apply mask overlay to visualization
    vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)
    
    # Add image info text
    info_text = [
        f"Image: {Path(image_path).name}",
        f"Detections: {len(result.boxes)}",
        f"ChArUco corners: {len(charuco_corners) if charuco_corners is not None else 0}",
        f"Confidence threshold: {confidence_threshold}",
        f"Model: YOLOv11 Segmentation"
    ]
    
    # Draw info background
    y_offset = 30
    max_info_width = max([cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for text in info_text])
    
    cv2.rectangle(vis_image, (10, 10), 
                 (10 + max_info_width + 20, y_offset + len(info_text)*25 + 10), 
                 (0, 0, 0), -1)
    cv2.rectangle(vis_image, (10, 10), 
                 (10 + max_info_width + 20, y_offset + len(info_text)*25 + 10), 
                 (255, 255, 255), 2)
    
    for i, text in enumerate(info_text):
        cv2.putText(vis_image, text, (15, y_offset + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_image, detection_info, charuco_corners is not None

def visualize_from_json_results(json_file="detection_results_segmentation.json", num_images=4, save_visualizations=True):
    """Visualize detections using results from JSON file"""
    
    # Load configuration
    model_path = "yolo_training_output/yolo11s-seg_frame_detector/weights/best.pt"  # Updated path
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
    
    print(f"=== Visualizing {min(num_images, len(results_to_show))} images from {json_file} ===\n")
    
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
                mask_info = f", mask: {det['mask_area_px']:,}px²" if det['mask_area_px'] > 0 else ""
                print(f"    {det['id']}: {det['class']} ({det['confidence']:.2f}) - {det['width_px']}×{det['height_px']}px{mask_info}")
            
            # Print measurements from JSON
            for det in result['detections']:
                if det['real_world_measurements']:
                    measurements = det['real_world_measurements']
                    if 'polygon_points_3d' in measurements:
                        # Segmentation-based measurements
                        print(f"    Real measurements (from mask): {measurements['area_m2']:.4f}m² "
                              f"(approx {measurements['width_m']:.2f}×{measurements['height_m']:.2f}m)")
                    else:
                        # Bbox-based measurements
                        print(f"    Real measurements (from bbox): {measurements['width_m']:.2f}m × {measurements['height_m']:.2f}m")
            print()
    
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
        model_path = "yolo_training_output/yolo11s-seg_frame_detector/weights/best.pt"  # Updated path
    
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
        plt.title(f"YOLOv11 Segmentation Detection: {Path(image_path).name}")
        plt.axis('off')
        plt.show()
        
        # Print details
        print(f"Image: {Path(image_path).name}")
        print(f"ChArUco board detected: {has_charuco}")
        print(f"Number of detections: {len(detection_info)}")
        
        for det in detection_info:
            mask_info = f", mask area: {det['mask_area_px']:,}px²" if det['mask_area_px'] > 0 else ""
            coverage_info = f" ({(det['mask_area_px']/det['bbox_area_px']*100):.1f}% coverage)" if det['mask_area_px'] > 0 else ""
            print(f"  {det['class']}: {det['confidence']:.2f} confidence, "
                  f"bbox: {det['width_px']}×{det['height_px']}px{mask_info}{coverage_info}")
    
    return vis_image

def compare_bbox_vs_mask_measurements(json_file="detection_results_segmentation.json"):
    """Compare bounding box vs mask-based measurements"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("=== Comparison: Bounding Box vs Segmentation Mask Measurements ===\n")
    
    bbox_measurements = []
    mask_measurements = []
    
    for result in data['results']:
        for det in result['detections']:
            if det['real_world_measurements'] and det['mask_data']:
                measurements = det['real_world_measurements']
                mask_data = det['mask_data']
                
                if 'polygon_points_3d' in measurements:
                    # Mask-based measurement
                    mask_measurements.append({
                        'image': result['image_name'],
                        'area_m2': measurements['area_m2'],
                        'width_m': measurements['width_m'],
                        'height_m': measurements['height_m'],
                        'mask_area_px': mask_data['mask_area_pixels'],
                        'bbox_area_px': det['bbox_pixels']['width_px'] * det['bbox_pixels']['height_px']
                    })
                else:
                    # Bbox-based measurement
                    bbox_measurements.append({
                        'image': result['image_name'],
                        'area_m2': measurements['area_m2'],
                        'width_m': measurements['width_m'],
                        'height_m': measurements['height_m']
                    })
    
    print(f"Found {len(mask_measurements)} mask-based measurements")
    print(f"Found {len(bbox_measurements)} bbox-based measurements")
    
    if mask_measurements:
        print("\nMask-based measurements:")
        for m in mask_measurements:
            coverage = (m['mask_area_px'] / m['bbox_area_px']) * 100
            print(f"  {m['image']}: {m['area_m2']:.4f}m² ({m['width_m']:.2f}×{m['height_m']:.2f}m) "
                  f"- {coverage:.1f}% bbox coverage")
    
    if bbox_measurements:
        print("\nBbox-based measurements:")
        for m in bbox_measurements:
            print(f"  {m['image']}: {m['area_m2']:.4f}m² ({m['width_m']:.2f}×{m['height_m']:.2f}m)")

if __name__ == "__main__":
    print("=== YOLOv11 Segmentation Detection + ChArUco Visualization ===")
    print()
    
    # Option 1: Visualize from JSON results
    print("1. Visualizing from JSON results...")
    try:
        visualize_from_json_results("detection_results_segmentation.json", num_images=4, save_visualizations=True)
    except FileNotFoundError:
        print("❌ detection_results_segmentation.json not found. Run 05_generate_json_output.py first.")
        print("Trying original filename...")
        try:
            visualize_from_json_results("detection_results.json", num_images=4, save_visualizations=True)
        except FileNotFoundError:
            print("❌ No JSON results found. Run 05_generate_json_output.py first.")
    except Exception as e:
        print(f"❌ Error visualizing from JSON: {e}")
    
    print()
    
    # Option 2: Compare measurements
    print("2. Comparing bbox vs mask measurements...")
    try:
        compare_bbox_vs_mask_measurements("detection_results_segmentation.json")
    except FileNotFoundError:
        try:
            compare_bbox_vs_mask_measurements("detection_results.json")
        except FileNotFoundError:
            print("❌ No JSON results found for comparison.")
    except Exception as e:
        print(f"❌ Error comparing measurements: {e}")
    
    print()
    
    # Option 3: Visualize a single image (example)
    print("3. Example single image visualization...")
    single_image_path = "s20plus_images_with_ChArUco/20250804_122919.jpg"  # Update this path
    
    if Path(single_image_path).exists():
        try:
            visualize_single_image(single_image_path)
        except Exception as e:
            print(f"❌ Error visualizing single image: {e}")
    else:
        print(f"❌ Image not found: {single_image_path}")
        print("Update the single_image_path variable to point to an existing image.")