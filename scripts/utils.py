import cv2
import yaml
import numpy as np
import os
import glob
from datetime import datetime
import config.settings as settings
from pathlib import Path


def count_images(directory):
    """Count jpg and png images in directory"""
    path = Path(directory)
    if not path.exists():
        return 0
    return len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))


def get_detector_params():
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.adaptiveThreshWinSizeMin = settings.ADAPTIVE_THRESH_WIN_SIZE_MIN
    detector_params.adaptiveThreshWinSizeMax = settings.ADAPTIVE_THRESH_WIN_SIZE_MAX
    detector_params.adaptiveThreshWinSizeStep = settings.ADAPTIVE_THRESH_WIN_SIZE_STEP
    detector_params.minMarkerPerimeterRate = settings.MIN_MARKER_PERIMETER_RATE
    detector_params.maxMarkerPerimeterRate = settings.MAX_MARKER_PERIMETER_RATE
    detector_params.cornerRefinementMethod = settings.CORNER_REFINEMENT_METHOD
    return detector_params


def create_charuco_board():
    return cv2.aruco.CharucoBoard(
        size=(settings.CHARUCOBOARD_COLCOUNT, settings.CHARUCOBOARD_ROWCOUNT),
        squareLength=settings.SQUARE_LENGTH,
        markerLength=settings.MARKER_LENGTH,
        dictionary=cv2.aruco.getPredefinedDictionary(settings.ARUCO_DICT_ID),
    )


def load_camera_calibration(calib_file=settings.CALIB_OUTPUT_FILE):
    """Load camera calibration parameters"""
    with open(calib_file, "r") as f:
        calib_data = yaml.load(f, Loader=yaml.FullLoader)  # Fixed YAML loading

    camera_matrix = np.array(calib_data["camera_matrix"])
    dist_coeffs = np.array(calib_data["dist_coeff"])

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_width": calib_data["image_width"],
        "image_height": calib_data["image_height"],
        "square_length_m": calib_data["square_length_m"],
        "marker_length_m": calib_data["marker_length_m"],
        "charuco_dict": calib_data["charuco_dict"],
    }


def letterbox_yolo_style(
    img, new_shape=settings.YOLO_INPUT_SHAPE, color=settings.LETTERBOX_COLOR
):
    """Apply YOLO letterboxing exactly as used in calibration"""

    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    scale_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute new dimensions
    new_unpad = int(round(shape[1] * scale_ratio)), int(round(shape[0] * scale_ratio))

    # Resize image
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Compute padding
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    # Divide padding into 2 sides
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    img_letterboxed = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return img_letterboxed, scale_ratio, (left, top)


def ensure_landscape_orientation(image):
    """
    Rotate image to landscape if it's portrait for consistent processing.
    Returns: (processed_image, was_rotated)
    """
    h, w = image.shape[:2]
    if h > w:  # Portrait orientation - rotate to landscape
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image, True
    return image, False


def adjust_measurements_for_rotation(measurements, was_rotated):
    """
    Swap width/height measurements if image was rotated from portrait to landscape.
    This ensures measurements match the original image orientation.
    """
    if not measurements or not was_rotated:
        return measurements

    # Swap width and height to match original portrait orientation
    if "width_cm" in measurements and "height_cm" in measurements:
        width_cm = measurements["width_cm"]
        height_cm = measurements["height_cm"]
        measurements["width_cm"] = (
            height_cm  # Original width becomes height after rotation
        )
        measurements["height_cm"] = (
            width_cm  # Original height becomes width after rotation
        )

    return measurements


def detect_charuco_board(
    image, aruco_dict, charuco_board, detector_params, visualize_corners_dir, img_path
):
    """Detect ChArUco board"""

    visualize_corners = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers with improved parameters
    corners, markers_ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=detector_params
    )

    if markers_ids is not None and len(markers_ids) >= settings.MIN_MARKERS_PER_IMAGE:
        # Interpolate ChArUco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=markers_ids,
            image=gray,
            board=charuco_board,
        )
        if (
            retval
            and charuco_corners is not None
            and charuco_ids is not None
            and len(charuco_ids) >= settings.MIN_CORNERS_PER_IMAGE
        ):

            # add print of markers, corners, scale
            cv2.aruco.drawDetectedCornersCharuco(
                visualize_corners, charuco_corners, charuco_ids, (255, 0, 0)
            )
            for i, (pt, cid) in enumerate(zip(charuco_corners, charuco_ids)):
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(visualize_corners, (x, y), 3, (255, 0, 0), -1)
                cv2.putText(
                    visualize_corners,
                    str(int(cid)),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            file_name = os.path.basename(img_path)
            # file_name = f"visualize_corners_{original_image_name}_{len(os.listdir(visualize_corners_dir)) + 1}"

            cv2.imwrite(
                os.path.join(visualize_corners_dir, f"{file_name}"),
                visualize_corners,
            )
        return markers_ids, charuco_corners, charuco_ids

    return None, None, None


def polygon_to_bbox(polygon):
    """Convert polygon points to bounding box [x1, y1, x2, y2]"""
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    return [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]


def calculate_polygon_area_pixels(polygon):
    """Calculate polygon area in pixels using the shoelace formula"""
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def project_points_to_board_plane(points_2d_px, calib, rvec, tvec):
    """
    points_2d_px: (N,2) float32/float64 pixel coords in the SAME (letterboxed) image used for PnP
    Returns: (N,3) array of [x,y,0] in board (object) frame, meters
    Skips points where the ray is parallel to the plane.
    """
    K = calib["camera_matrix"]
    dist = calib["dist_coeffs"]

    # 1) undistort + normalize
    pts = points_2d_px.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(
        pts, K, dist
    )  # -> shape (N,1,2), normalized camera coords
    undist = undist.reshape(-1, 2)

    # 2) camera->object transforms
    R, _ = cv2.Rodrigues(rvec)  # camera <- object
    Rt = R.T  # object <- camera
    t = tvec.reshape(3, 1)
    C_obj = -Rt @ t  # camera center in object frame

    out = []
    for i in range(len(undist)):
        x, y = undist[i]
        d_cam = np.array([[x], [y], [1.0]])  # ray dir in camera frame (Z=1 convention)
        d_obj = Rt @ d_cam  # ray dir in object frame

        denom = d_obj[2, 0]  # Z component in object frame
        if abs(denom) < 1e-9:
            # ray nearly parallel to plane Z=0
            continue

        s = -C_obj[2, 0] / denom  # intersection scale
        X_obj = C_obj + s * d_obj  # (x,y,0)
        if abs(X_obj[2, 0]) > 1e-6:
            # numerical safety; should be ~0
            continue

        out.append([float(X_obj[0, 0]), float(X_obj[1, 0]), 0.0])

    if not out:
        return None
    return np.array(out, dtype=float)


def is_pose_reasonable(rvec, tvec):
    """Check if pose estimation is reasonable for indoor photography with more permissive ranges"""
    try:
        distance = abs(tvec[2].item())
        rotation_magnitude = np.linalg.norm(rvec)

        # More permissive ranges for indoor photography with ChArUco board
        distance_ok = 0.2 < distance < 5.0  # 20cm to 5m (more permissive)
        rotation_ok = (
            rotation_magnitude < 2.5
        )  # Less than ~143 degrees (more permissive)

        return distance_ok and rotation_ok, distance, rotation_magnitude
    except:
        return False, 0, 0


def calculate_real_world_measurements_polygon(
    polygon_pixels, charuco_corners, charuco_ids, charuco_board, calib_data
):
    """Calculate real-world dimensions using polygon and ChArUco board reference with improved robustness"""

    if (
        charuco_corners is None or len(charuco_corners) < 4
    ):  # More permissive requirement
        return None

    # Solve PnP with RANSAC for robustness
    object_points = charuco_board.getChessboardCorners()[charuco_ids.flatten()]

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=charuco_corners,
            cameraMatrix=calib_data["camera_matrix"],
            distCoeffs=calib_data["dist_coeffs"],
            iterationsCount=1000,
            reprojectionError=3.0,  # More permissive - increased from 2.0
            confidence=0.95,  # More permissive - reduced from 0.99
        )
    except:
        # Fallback to regular solvePnP if RANSAC fails
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=charuco_corners,
            cameraMatrix=calib_data["camera_matrix"],
            distCoeffs=calib_data["dist_coeffs"],
        )
        inliers = None

    if success:
        # Check if pose is reasonable with more permissive ranges
        pose_ok, distance, rotation = is_pose_reasonable(rvec, tvec)
        inlier_count = len(inliers) if inliers is not None else len(charuco_corners)

        print(
            f"    DEBUG: tvec[2]={tvec[2].item():.3f}m, rotation={rotation:.2f}rad, inliers={inlier_count}/{len(charuco_corners)}"
        )

        if not pose_ok:
            print(
                f"    WARNING: Unrealistic pose (dist={distance:.1f}m, rot={rotation:.2f}rad), skipping"
            )
            return None

        # More permissive inlier requirement
        if inliers is not None and len(inliers) < max(
            3, len(charuco_corners) * 0.4
        ):  # At least 40% or 3 points
            print(
                f"    WARNING: Too few inliers ({len(inliers)}/{len(charuco_corners)}), pose unreliable"
            )
            return None
    else:
        print(f"    ERROR: PnP solve failed")
        return None

    # Project polygon corners to 3D plane (z=0, same plane as ChArUco board)
    polygon_corners_3d = project_points_to_board_plane(
        polygon_pixels.astype(np.float64), calib_data, rvec, tvec
    )

    if polygon_corners_3d is None or len(polygon_corners_3d) < 3:
        print("    WARNING: Not enough projected polygon points")
        return None

    polygon_corners_3d = np.array(polygon_corners_3d, dtype=float)

    try:
        # Calculate polygon area in 3D (projected to 2D since z=0)
        polygon_2d = polygon_corners_3d[:, :2]  # Take only x, y coordinates
        area_m2 = calculate_polygon_area_pixels(polygon_2d)

        # Calculate approximate width and height from bounding box
        x_coords = polygon_2d[:, 0]
        y_coords = polygon_2d[:, 1]
        width_m = np.max(x_coords) - np.min(x_coords)
        height_m = np.max(y_coords) - np.min(y_coords)

        # Calculate perimeter
        perimeter_m = 0.0
        for i in range(len(polygon_2d)):
            p1 = polygon_2d[i]
            p2 = polygon_2d[(i + 1) % len(polygon_2d)]
            perimeter_m += np.linalg.norm(p2 - p1)

        # More permissive sanity check on measurements
        if width_m > 15 or height_m > 15 or area_m2 > 200:  # Increased limits
            print(
                f"    WARNING: Measurements seem too large (w={width_m:.2f}m, h={height_m:.2f}m, a={area_m2:.2f}m²)"
            )
            return None

        if width_m < 0.005 or height_m < 0.005:  # Smaller than 0.5cm
            print(
                f"    WARNING: Measurements seem too small (w={width_m:.4f}m, h={height_m:.4f}m)"
            )
            return None

        # Debug output
        print(
            f"    DEBUG: area_m2={area_m2:.6f}, width_m={width_m:.3f}, height_m={height_m:.3f}, perimeter_m={perimeter_m:.3f}"
        )

    except Exception as e:
        print(f"    ERROR calculating polygon dimensions: {e}")
        return None

    # Convert to centimeters
    return {
        "area_cm2": float(area_m2 * 10000),  # m² to cm² (×10000)
        "width_cm": float(width_m * 100),  # m to cm (×100)
        "height_cm": float(height_m * 100),  # m to cm (×100)
        "perimeter_cm": float(perimeter_m * 100),  # m to cm (×100)
        "polygon_points_3d": polygon_corners_3d.tolist(),
        "projected_points_count": len(polygon_corners_3d),
    }


def calculate_real_world_measurements_bbox(
    bbox_pixels, charuco_corners, charuco_ids, charuco_board, calib_data
):
    """Calculate real-world dimensions using bounding box and ChArUco board reference with improved robustness"""

    if (
        charuco_corners is None or len(charuco_corners) < 4
    ):  # More permissive requirement
        return None

    # Solve PnP with RANSAC for robustness
    object_points = charuco_board.getChessboardCorners()[charuco_ids.flatten()]

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=charuco_corners,
            cameraMatrix=calib_data["camera_matrix"],
            distCoeffs=calib_data["dist_coeffs"],
            iterationsCount=1000,
            reprojectionError=3.0,  # More permissive
            confidence=0.95,  # More permissive
        )
    except:
        # Fallback to regular solvePnP if RANSAC fails
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=charuco_corners,
            cameraMatrix=calib_data["camera_matrix"],
            distCoeffs=calib_data["dist_coeffs"],
        )
        inliers = None

    if success:
        # Check if pose is reasonable
        pose_ok, distance, rotation = is_pose_reasonable(rvec, tvec)
        print(f"    DEBUG: tvec[2]={tvec[2].item():.3f}m, rotation={rotation:.2f}rad")

        if not pose_ok:
            print(
                f"    WARNING: Unrealistic pose (dist={distance:.1f}m, rot={rotation:.2f}rad), skipping"
            )
            return None

        # More permissive inlier requirement
        if inliers is not None and len(inliers) < max(3, len(charuco_corners) * 0.4):
            print(f"    WARNING: Too few inliers, pose might be unreliable")
            return None
    else:
        return None

    # Convert bounding box corners to 3D plane
    bbox_corners_2d = np.array(
        [
            [bbox_pixels[0], bbox_pixels[1]],  # top-left
            [bbox_pixels[2], bbox_pixels[1]],  # top-right
            [bbox_pixels[2], bbox_pixels[3]],  # bottom-right
            [bbox_pixels[0], bbox_pixels[3]],  # bottom-left
        ],
        dtype=np.float32,
    )

    bbox_corners_3d = project_points_to_board_plane(
        bbox_corners_2d, calib_data, rvec, tvec
    )
    if bbox_corners_3d is None or len(bbox_corners_3d) < 4:
        print("    WARNING: Not enough projected bbox corners")
        return None

    bbox_corners_3d = np.array(bbox_corners_3d, dtype=float)

    try:
        width_m = np.linalg.norm(bbox_corners_3d[1] - bbox_corners_3d[0])
        height_m = np.linalg.norm(bbox_corners_3d[3] - bbox_corners_3d[0])

        # More permissive sanity check on measurements
        if width_m > 15 or height_m > 15:  # Increased from 10m
            print(
                f"    WARNING: BBox measurements too large (w={width_m:.2f}m, h={height_m:.2f}m)"
            )
            return None

        if width_m < 0.005 or height_m < 0.005:  # Smaller than 0.5cm
            print(
                f"    WARNING: BBox measurements too small (w={width_m:.4f}m, h={height_m:.4f}m)"
            )
            return None

    except Exception as e:
        print(f"    ERROR calculating bbox dimensions: {e}")
        return None

    # Convert to centimeters
    return {
        "width_cm": float(width_m * 100),  # m to cm (×100)
        "height_cm": float(height_m * 100),  # m to cm (×100)
        "area_cm2": float(width_m * height_m * 10000),  # m² to cm² (×10000)
    }


def save_letterboxed_images_preview(input_dir, num_preview=3):
    """
    Save a few letterboxed images as preview to verify the preprocessing
    """
    preview_dir = "letterbox_preview"
    os.makedirs(preview_dir, exist_ok=True)

    images = glob.glob(os.path.join(input_dir, "*.jpg"))
    images.extend(glob.glob(os.path.join(input_dir, "*.png")))

    print(
        f"\nSaving {min(num_preview, len(images))} letterboxed preview images to {preview_dir}/"
    )

    for i, img_path in enumerate(images[:num_preview]):
        image = cv2.imread(img_path)
        if image is None:
            continue

        original_shape = image.shape[:2]
        letterboxed, scale_ratio, padding = letterbox_yolo_style(
            image,
            (settings.YOLO_INPUT_SIZE, settings.YOLO_INPUT_SIZE),
            settings.LETTERBOX_COLOR,
        )

        # Create side-by-side comparison
        # Resize original for display
        display_original = cv2.resize(image, (320, 240))
        display_letterboxed = cv2.resize(letterboxed, (320, 320))

        # Create comparison image
        comparison = np.zeros((max(240, 320), 320 + 320 + 20, 3), dtype=np.uint8)
        comparison[:240, :320] = display_original
        comparison[:320, 340:660] = display_letterboxed

        # Add text labels
        cv2.putText(
            comparison,
            f"Original: {original_shape[1]}x{original_shape[0]}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            comparison,
            f"Letterboxed: {settings.YOLO_INPUT_SIZE}x{settings.YOLO_INPUT_SIZE}",
            (350, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            comparison,
            f"Scale: {scale_ratio:.3f}",
            (350, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            comparison,
            f"Padding: {padding[0]:.1f}x{padding[1]:.1f}",
            (350, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Save comparison
        preview_path = os.path.join(
            preview_dir, f"preview_{i+1}_{os.path.basename(img_path)}"
        )
        cv2.imwrite(preview_path, comparison)
        print(f"  Saved: {preview_path}")


def process_s20plus_image(image_path, model, calib_data, confidence_threshold=0.25):
    """Process a single S20+ image with ChArUco board using consistent letterboxing and orientation handling"""

    # Load original image
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        return None

    # Store original dimensions for JSON output
    original_h, original_w = original_image.shape[:2]

    # Ensure landscape orientation for consistent processing
    landscape_image, was_rotated = ensure_landscape_orientation(original_image)

    # Apply SAME letterboxing as used in calibration (on landscape-oriented image)
    letterboxed_image, scale_ratio, padding = letterbox_yolo_style(
        landscape_image, settings.YOLO_INPUT_SHAPE, settings.LETTERBOX_COLOR
    )

    print(
        f"    Applied letterboxing: scale={scale_ratio:.3f}, padding=({padding[0]:.1f}, {padding[1]:.1f}), rotated={was_rotated}"
    )

    # YOLO detection on letterboxed image
    results = model.predict(
        source=letterboxed_image, conf=confidence_threshold, verbose=False
    )
    result = results[0]

    # Detect ChArUco board on letterboxed image (same as calibration)
    charuco_corners, charuco_ids, charuco_board = detect_charuco_board(
        letterboxed_image, calib_data
    )

    detections = []

    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2] in letterboxed coordinates

        detection = {
            "detection_id": i,
            "class_id": class_id,
            "class_name": model.names[class_id],
            "confidence": confidence,
            "bbox_pixels": {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
                "width_px": float(bbox[2] - bbox[0]),
                "height_px": float(bbox[3] - bbox[1]),
            },
            "charuco_detected": charuco_corners is not None,
            "real_world_measurements": None,
            "mask_data": None,
        }

        # Process segmentation mask if available
        if result.masks is not None and i < len(result.masks.data):
            mask = result.masks.data[i].cpu().numpy()

            # Resize mask to match letterboxed image dimensions if needed
            if mask.shape != letterboxed_image.shape[:2]:
                mask = cv2.resize(
                    mask, (letterboxed_image.shape[1], letterboxed_image.shape[0])
                )

            # Extract polygon from mask
            contours, _ = cv2.findContours(
                (mask > 0.5).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Simplify contour to reduce points
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                # Convert to list of points
                polygon_points = simplified_contour.reshape(-1, 2).astype(float)

                detection["mask_data"] = {
                    "polygon_points": polygon_points.tolist(),
                    "mask_area_pixels": float(
                        calculate_polygon_area_pixels(polygon_points)
                    ),
                    "mask_perimeter_pixels": float(
                        cv2.arcLength(largest_contour, True)
                    ),
                }

                # Calculate real-world measurements from polygon if ChArUco board is detected
                if charuco_corners is not None:
                    polygon_measurements = calculate_real_world_measurements_polygon(
                        polygon_points,
                        charuco_corners,
                        charuco_ids,
                        charuco_board,
                        calib_data,
                    )
                    if polygon_measurements:
                        # Adjust measurements for rotation to match original image orientation
                        polygon_measurements = adjust_measurements_for_rotation(
                            polygon_measurements, was_rotated
                        )
                        detection["real_world_measurements"] = polygon_measurements

        # Fall back to bbox measurements if no mask or mask processing failed
        if detection["real_world_measurements"] is None and charuco_corners is not None:
            bbox_measurements = calculate_real_world_measurements_bbox(
                bbox, charuco_corners, charuco_ids, charuco_board, calib_data
            )
            if bbox_measurements:
                # Adjust measurements for rotation to match original image orientation
                bbox_measurements = adjust_measurements_for_rotation(
                    bbox_measurements, was_rotated
                )
                detection["real_world_measurements"] = bbox_measurements

        detections.append(detection)

    return {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "timestamp": datetime.now().isoformat(),
        "image_dimensions": {
            "width": original_w,  # Report ORIGINAL dimensions
            "height": original_h,
        },
        "processing_info": {
            "was_rotated_for_processing": was_rotated,
            "letterbox_scale_ratio": scale_ratio,
            "letterbox_padding": padding,
            "processed_dimensions": {
                "width": letterboxed_image.shape[1],
                "height": letterboxed_image.shape[0],
            },
        },
        "charuco_board_detected": charuco_corners is not None,
        "charuco_corners_count": (
            len(charuco_corners) if charuco_corners is not None else 0
        ),
        "total_detections": len(detections),
        "detections": detections,
    }
