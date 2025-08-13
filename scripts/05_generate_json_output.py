from ultralytics import YOLO
import cv2
import numpy as np
import yaml
import json
import os
from pathlib import Path
from datetime import datetime

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

def detect_charuco_board(image, calib_data):
    """Detect ChArUco board in image and return corners"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(calib_data['charuco_dict'])
    charuco_board = cv2.aruco.CharucoBoard(
        size=(5, 5),  # 5x5 board
        squareLength=calib_data['square_length_m'],
        markerLength=calib_data['marker_length_m'],
        dictionary=aruco_dict
    )
    
    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    
    if ids is not None and len(ids) > 4:
        # Interpolate ChArUco corners
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=charuco_board
        )
        
        if charuco_corners is not None and len(charuco_corners) > 4:
            return charuco_corners, charuco_ids, charuco_board
    
    return None, None, None

def calculate_real_world_measurements(bbox_pixels, charuco_corners, charuco_ids, charuco_board, calib_data):
    """Calculate real-world dimensions using ChArUco board reference"""
    
    if charuco_corners is None or len(charuco_corners) < 4:
        return None
    
    # Solve PnP to get pose
    object_points = charuco_board.getChessboardCorners()[charuco_ids.flatten()]
    
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=charuco_corners,
        cameraMatrix=calib_data['camera_matrix'],
        distCoeffs=calib_data['dist_coeffs']
    )
    
    if not success:
        return None
    
    # Convert bounding box corners to 3D plane
    bbox_corners_2d = np.array([
        [bbox_pixels[0], bbox_pixels[1]],  # top-left
        [bbox_pixels[2], bbox_pixels[1]],  # top-right
        [bbox_pixels[2], bbox_pixels[3]],  # bottom-right
        [bbox_pixels[0], bbox_pixels[3]]   # bottom-left
    ], dtype=np.float32)
    
    # Project bbox corners to 3D plane (z=0, same plane as ChArUco board)
    bbox_corners_3d = []
    for corner_2d in bbox_corners_2d:
        # Unproject to ray
        corner_normalized = cv2.undistortPoints(
            corner_2d.reshape(1, 1, 2),
            calib_data['camera_matrix'],
            calib_data['dist_coeffs']
        )[0][0]
        
        # Find intersection with z=0 plane (ChArUco board plane)
        # This is a simplified approach - assumes board is roughly parallel to detection plane
        scale = -tvec[2] / corner_normalized[1] if corner_normalized[1] != 0 else 1
        x_3d = corner_normalized[0] * scale + tvec[0]
        y_3d = corner_normalized[1] * scale + tvec[1]
        z_3d = 0  # On the board plane
        
        bbox_corners_3d.append([x_3d, y_3d, z_3d])
    
    # bbox_corners_3d = np.array(bbox_corners_3d)
    # Ensure numpy array and handle polygons with more than 4 points
    bbox_corners_3d = np.array(bbox_corners_3d, dtype=float)

    if bbox_corners_3d.shape[0] != 4:
        # Compute bounding rectangle from all points
        min_x, min_y, min_z = bbox_corners_3d.min(axis=0)
        max_x, max_y, max_z = bbox_corners_3d.max(axis=0)
        bbox_corners_3d = np.array([
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z]
        ], dtype=float)


    # Calculate width and height in meters
    width_m = np.linalg.norm(bbox_corners_3d[1] - bbox_corners_3d[0])
    height_m = np.linalg.norm(bbox_corners_3d[3] - bbox_corners_3d[0])
    
    return {
        'width_m': float(width_m),
        'height_m': float(height_m),
        'area_m2': float(width_m * height_m)
    }

def process_s20plus_image(image_path, model, calib_data, confidence_threshold=0.25):
    """Process a single S20+ image with ChArUco board"""
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # YOLO detection
    results = model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)
    result = results[0]
    
    # Detect ChArUco board
    charuco_corners, charuco_ids, charuco_board = detect_charuco_board(image, calib_data)
    
    detections = []
    
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        
        detection = {
            'detection_id': i,
            'class_id': class_id,
            'class_name': 'window' if class_id == 0 else 'door',
            'confidence': confidence,
            'bbox_pixels': {
                'x1': float(bbox[0]),
                'y1': float(bbox[1]),
                'x2': float(bbox[2]),
                'y2': float(bbox[3]),
                'width_px': float(bbox[2] - bbox[0]),
                'height_px': float(bbox[3] - bbox[1])
            },
            'charuco_detected': charuco_corners is not None,
            'real_world_measurements': None
        }
        
        # Calculate real-world measurements if ChArUco board is detected
        if charuco_corners is not None:
            measurements = calculate_real_world_measurements(
                bbox, charuco_corners, charuco_ids, charuco_board, calib_data
            )
            detection['real_world_measurements'] = measurements
        
        detections.append(detection)
    
    return {
        'image_path': str(image_path),
        'image_name': image_path.name,
        'timestamp': datetime.now().isoformat(),
        'image_dimensions': {
            'width': image.shape[1],
            'height': image.shape[0]
        },
        'charuco_board_detected': charuco_corners is not None,
        'charuco_corners_count': len(charuco_corners) if charuco_corners is not None else 0,
        'total_detections': len(detections),
        'detections': detections
    }

def generate_json_output():
    """Main function to process S20+ images and generate JSON output"""
    
    # === CONFIGURATION ===
    model_path = "yolo_training_output/yolov8n_window_door_detector/weights/best.pt"
    s20plus_images_dir = "s20plus_images_with_ChArUco"
    calib_file = "config/s20plus_calib.yaml"
    output_file = "detection_results.json"
    confidence_threshold = 0.25
    
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
    
    print("Loading YOLO model...")
    model = YOLO(model_path)
    
    print("Loading camera calibration...")
    calib_data = load_camera_calibration(calib_file)
    
    # === GET S20+ IMAGES ===
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
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
        'total_images': len(s20plus_images),
        'images_with_charuco': 0,
        'images_with_detections': 0,
        'total_detections': 0,
        'measurements_calculated': 0
    }
    
    for i, img_path in enumerate(s20plus_images):
        print(f"Processing {i+1}/{len(s20plus_images)}: {img_path.name}")
        
        result = process_s20plus_image(img_path, model, calib_data, confidence_threshold)
        
        if result:
            all_results.append(result)
            
            # Update statistics
            if result['charuco_board_detected']:
                stats['images_with_charuco'] += 1
            
            if result['total_detections'] > 0:
                stats['images_with_detections'] += 1
                stats['total_detections'] += result['total_detections']
                
                # Count measurements
                for detection in result['detections']:
                    if detection['real_world_measurements'] is not None:
                        stats['measurements_calculated'] += 1
            
            print(f"  Detections: {result['total_detections']}, ChArUco: {'Yes' if result['charuco_board_detected'] else 'No'}")
    
    # === SAVE RESULTS ===
    output_data = {
        'metadata': {
            'generation_timestamp': datetime.now().isoformat(),
            'model_used': model_path,
            'calibration_file': calib_file,
            'confidence_threshold': confidence_threshold,
            'total_images_processed': len(all_results)
        },
        'statistics': stats,
        'results': all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # === SUMMARY ===
    print("\n" + "="*50)
    print("JSON GENERATION SUMMARY")
    print("="*50)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with ChArUco board: {stats['images_with_charuco']}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Real-world measurements calculated: {stats['measurements_calculated']}")
    print(f"JSON output saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    try:
        output_file = generate_json_output()
        print(f"\n✅ JSON output generated successfully: {output_file}")
        
        # Optionally load and display a sample
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        print("\nSample detection (first result with measurements):")
        for result in data['results']:
            for detection in result['detections']:
                if detection['real_world_measurements']:
                    print(f"  {detection['class_name']}: {detection['real_world_measurements']['width_m']:.3f}m × {detection['real_world_measurements']['height_m']:.3f}m")
                    break
            else:
                continue
            break
        
    except Exception as e:
        print(f"❌ Error generating JSON output: {e}")
        raise