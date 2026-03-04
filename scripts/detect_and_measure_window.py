import json
import time
import cv2
import os
from pathlib import Path
import numpy as np
import scripts.utils as utils
import config.settings as settings


class OptimalMeasurementSystem:
    """
    Most accurate measurement system using proper geometric analysis of segmentation masks.
    """

    def __init__(self):
        """Initialize the window measurement system."""
        # Use settings for ChArUco board specifications
        self.board_cols = settings.CHARUCOBOARD_COLCOUNT
        self.board_rows = settings.CHARUCOBOARD_ROWCOUNT
        self.square_size_mm = settings.SQUARE_LENGTH * 1000  # Convert from meters to mm
        self.marker_size_mm = settings.MARKER_LENGTH * 1000  # Convert from meters to mm

        # Use existing ChArUco board from utils
        self.charuco_board = utils.create_charuco_board()

        # Use existing detector parameters from utils
        self.detector_params = utils.get_detector_params()

        # Load YOLO model using existing utils function
        self.detection_dataset_dir = settings.SUPER_TESTER_PATH
        self.yolo_model, self.model_path = utils.load_model_and_validate_paths(
            self.detection_dataset_dir
        )

        # Measurement accuracy target
        self.target_accuracy_mm = 10

    def calculate_pixel_to_mm_ratio(self, charuco_corners, charuco_ids):
        if charuco_corners is None or len(charuco_corners) < 2:
            return None, False

        pixel_distances = []

        # Create a mapping from corner ID to pixel position
        id_to_corner = {}
        for i, corner_id in enumerate(charuco_ids):
            id_to_corner[corner_id[0]] = charuco_corners[i][0]

        detected_ids = [cid[0] for cid in charuco_ids]

        # Check all pairs of detected corners for adjacency
        for i, id1 in enumerate(detected_ids):
            for j, id2 in enumerate(detected_ids):
                if i >= j:  # Skip duplicate pairs and self-comparison
                    continue

                # Calculate grid positions
                row1, col1 = divmod(id1, self.board_cols - 1)
                row2, col2 = divmod(id2, self.board_cols - 1)

                # Check if adjacent
                if (abs(row1 - row2) == 1 and col1 == col2) or (
                    abs(col1 - col2) == 1 and row1 == row2
                ):
                    # Calculate pixel distance
                    p1 = id_to_corner[id1]
                    p2 = id_to_corner[id2]
                    pixel_dist = np.linalg.norm(p1 - p2)
                    # print(
                    #     f"Pixel distance between {id1} and {id2}: {pixel_dist:.2f} pixels"
                    # )
                    pixel_distances.append(pixel_dist)

        if len(pixel_distances) == 0:
            return None, False

        # Calculate average pixels per mm
        pixels_per_mm = np.mean(pixel_distances) / self.square_size_mm
        print(f"Pixels per mm: {pixels_per_mm:.2f}")
        return pixels_per_mm, True

    def detect_windows_with_optimal_accuracy(self, image_path):
        """
        Detect windows and extract masks for precise geometric analysis.
        """
        result, inference_time, image, masks, mask_info = (
            utils.process_single_image_with_mask_extraction(
                self.yolo_model,
                image_path,
                settings.CONFIDENCE_THRESHOLD,
                settings.IOU_THRESHOLD,
            )
        )

        if result is None:
            return [], 0, image

        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            for i in range(len(boxes)):
                confidence = float(boxes.conf[i])

                if masks is not None and i < len(masks):
                    mask = masks[i]

                    # Perform comprehensive mask analysis
                    mask_analysis = self.analyze_mask_geometry(mask, image.shape[:2])

                    detection = {
                        "id": i + 1,
                        "confidence": confidence,
                        "mask_analysis": mask_analysis,
                        "measurement_method": "optimal_geometric_analysis",
                    }
                    detections.append(detection)

        print(
            f"Detected {len(detections)} windows with optimal analysis in {inference_time:.3f}s"
        )
        return detections, inference_time, image

    def analyze_mask_geometry(self, mask, original_shape):
        """
        Comprehensive geometric analysis of segmentation mask.
        Returns multiple measurement approaches for cross-validation.
        """
        # Resize mask to original image size
        mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]))
        binary_mask = (mask_resized > 0.5).astype(np.uint8)

        analysis = {}

        # Method 1: Minimum Area Rectangle (Most reliable for rectangular objects)
        analysis["min_area_rect"] = self.get_minimum_area_rectangle(binary_mask)

        # Method 2: Contour-based analysis (Most precise boundary detection)
        analysis["contour_analysis"] = self.get_contour_dimensions(binary_mask)

        # Quality metrics
        analysis["quality_metrics"] = self.calculate_measurement_quality(
            binary_mask, analysis
        )

        return analysis

    def get_minimum_area_rectangle(self, binary_mask):
        """
        Method 1: Calculate minimum area rectangle - most reliable for rectangular objects.
        """
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {
                "width": 0,
                "height": 0,
                "angle": 0,
                "confidence": 0,
                "coordinates": [],
            }

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        (center_x, center_y), (width, height), angle = rect

        # Get corner coordinates
        box_points = cv2.boxPoints(rect)

        # Ensure width > height by convention
        if width < height:
            width, height = height, width
            angle = angle + 90

        return {
            "width": float(width),
            "height": float(height),
            "angle": float(angle),
            "center": (float(center_x), float(center_y)),
            "coordinates": box_points.tolist(),
            "confidence": 1.0,  # MinAreaRect is always reliable
            "method": "minimum_area_rectangle",
        }

    def get_contour_dimensions(self, binary_mask):
        """
        Method 3: Analyze contour for precise boundary measurements.
        """
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {"width": 0, "height": 0, "confidence": 0}

        largest_contour = max(contours, key=cv2.contourArea)

        # Simplify contour to reduce noise
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # For rectangular objects, try to fit a rectangle
        if len(simplified_contour) >= 4:
            # Use minimum area rectangle on simplified contour
            rect = cv2.minAreaRect(simplified_contour)
            (center_x, center_y), (width, height), angle = rect

            # Calculate how well the contour fits a rectangle
            rect_area = width * height
            contour_area = cv2.contourArea(largest_contour)
            rectangularity = contour_area / rect_area if rect_area > 0 else 0

            return {
                "width": float(max(width, height)),  # Larger dimension as width
                "height": float(min(width, height)),  # Smaller dimension as height
                "angle": float(angle),
                "confidence": float(rectangularity),
                "contour_points": len(simplified_contour),
                "method": "contour_analysis",
            }

        return {"width": 0, "height": 0, "confidence": 0}

    def calculate_measurement_quality(self, binary_mask, analysis):
        """
        Calculate overall quality metrics for the measurements.
        """
        # Basic mask quality
        mask_area = np.sum(binary_mask > 0)

        # Calculate perimeter safely
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_perimeter = cv2.arcLength(largest_contour, True)
        else:
            mask_perimeter = 0

        # Rectangularity (how rectangular the shape is)
        if mask_area > 0 and mask_perimeter > 0:
            rectangularity = (4 * np.pi * mask_area) / (mask_perimeter**2)
            rectangularity = min(1.0, rectangularity)  # Normalize
        else:
            rectangularity = 0

        # Consistency between methods
        methods = [
            "min_area_rect",
            "contour_analysis",
        ]
        widths = []
        heights = []

        for method in methods:
            if method in analysis and analysis[method].get("width", 0) > 0:
                widths.append(analysis[method]["width"])
                heights.append(analysis[method]["height"])

        # Calculate coefficient of variation for consistency
        width_consistency = (
            1 - (np.std(widths) / np.mean(widths))
            if len(widths) > 1 and np.mean(widths) > 0
            else 0
        )
        height_consistency = (
            1 - (np.std(heights) / np.mean(heights))
            if len(heights) > 1 and np.mean(heights) > 0
            else 0
        )

        overall_quality = (rectangularity + width_consistency + height_consistency) / 3

        return {
            "mask_area": int(mask_area),
            "mask_perimeter": float(mask_perimeter),
            "rectangularity": float(rectangularity),
            "width_consistency": float(width_consistency),
            "height_consistency": float(height_consistency),
            "overall_quality": float(overall_quality),
            "measurement_methods_count": len(widths),
        }

    def get_best_measurement(self, mask_analysis, pixels_per_mm):
        """
        Select the most reliable measurement from all methods.
        """
        methods = [
            "min_area_rect",
            "contour_analysis",
        ]

        # Weight each method based on reliability and confidence
        method_weights = {
            "min_area_rect": 0.7,  # Most reliable for rectangular objects
            "contour_analysis": 0.3,  # Good for boundary precision
        }

        weighted_width = 0
        weighted_height = 0
        total_weight = 0

        valid_measurements = []

        for method in methods:
            if method in mask_analysis:
                data = mask_analysis[method]
                if data.get("width", 0) > 0 and data.get("height", 0) > 0:
                    method_confidence = data.get("confidence", 0.5)
                    base_weight = method_weights[method]
                    final_weight = base_weight * method_confidence

                    weighted_width += data["width"] * final_weight
                    weighted_height += data["height"] * final_weight
                    total_weight += final_weight

                    valid_measurements.append(
                        {
                            "method": method,
                            "width_mm": data["width"] / pixels_per_mm,
                            "height_mm": data["height"] / pixels_per_mm,
                            "confidence": method_confidence,
                            "weight": final_weight,
                        }
                    )

        if total_weight > 0:
            final_width_pixels = weighted_width / total_weight
            final_height_pixels = weighted_height / total_weight

            final_width_mm = final_width_pixels / pixels_per_mm
            final_height_mm = final_height_pixels / pixels_per_mm

            # Overall confidence based on quality metrics
            overall_confidence = mask_analysis["quality_metrics"]["overall_quality"]

            return {
                "width_mm": float(final_width_mm),
                "height_mm": float(final_height_mm),
                "confidence": float(overall_confidence),
                "valid_methods": len(valid_measurements),
                "method_details": valid_measurements,
                "measurement_method": "weighted_multi_method",
            }

        return None

    def save_annotated_image_with_masks(self, image, results, detections, output_dir):
        """Save image with mask overlays and measurements."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            annotated_image = image.copy()

            # Draw mask contours and measurements
            for i, window in enumerate(results["windows"]):
                # Get the corresponding detection for mask info
                detection = detections[i] if i < len(detections) else None

                # Draw mask contour if available
                if detection and "mask_analysis" in detection:
                    min_area_rect = detection["mask_analysis"].get("min_area_rect", {})
                    if "coordinates" in min_area_rect and min_area_rect["coordinates"]:
                        contour_coords = np.array(
                            min_area_rect["coordinates"], dtype=np.int32
                        )
                        cv2.polylines(
                            annotated_image, [contour_coords], True, (255, 0, 0), 2
                        )  # Blue contour

                # Draw oriented bounding box (if coordinates available)
                if "coordinates" in window:
                    coords = np.array(window["coordinates"]).reshape(4, 2).astype(int)
                    cv2.polylines(
                        annotated_image, [coords], True, (0, 255, 0), 2
                    )  # Green OBB
                    center = coords.mean(axis=0).astype(int)
                else:
                    # Fallback: use min_area_rect center if no coordinates
                    if detection and "mask_analysis" in detection:
                        min_area_rect = detection["mask_analysis"].get(
                            "min_area_rect", {}
                        )
                        if "center" in min_area_rect:
                            center = np.array(min_area_rect["center"], dtype=int)
                        else:
                            center = np.array([100, 100])  # Default position
                    else:
                        center = np.array([100, 100])

                # Add measurements and confidence
                text1 = f"W:{window['width_mm']:.0f}mm H:{window['height_mm']:.0f}mm"
                text2 = f"Conf:{window.get('confidence', window.get('detection_confidence', 0)):.2f}"
                text3 = f"Methods:{window.get('valid_methods', 0)}"

                # Add background for text readability
                text_size1 = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    annotated_image,
                    (center[0] - 70, center[1] - 40),
                    (center[0] + text_size1[0] + 10, center[1] + 20),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    annotated_image,
                    text1,
                    (center[0] - 60, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated_image,
                    text2,
                    (center[0] - 40, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    annotated_image,
                    text3,
                    (center[0] - 30, center[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

            # Save annotated image
            filename = f"optimal_measured_{Path(results['image_path']).name}"
            output_path = os.path.join(output_dir, filename)
            success = cv2.imwrite(output_path, annotated_image)

            if success:
                print(f"Mask-annotated image saved: {output_path}")
            else:
                print(f"Failed to save annotated image: {output_path}")

        except Exception as e:
            print(f"Error saving annotated image: {str(e)}")
            import traceback

            traceback.print_exc()

    def process_image_with_optimal_accuracy(self, image_path, output_dir=None):
        """
        Process image using the most accurate measurement approach.
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        print(f"Processing with optimal accuracy: {Path(image_path).name}")

        # ChArUco calibration (same as before)
        vis_dir = os.path.join(output_dir or "temp_measurement", "charuco_detection")
        os.makedirs(vis_dir, exist_ok=True)

        marker_ids, charuco_corners, charuco_ids = utils.detect_charuco_board(
            image,
            settings.ARUCO_DICT,
            self.charuco_board,
            self.detector_params,
            vis_dir,
            image_path,
        )

        charuco_success = (
            marker_ids is not None
            and charuco_corners is not None
            and charuco_ids is not None
        )

        if not charuco_success:
            return {
                "success": False,
                "error": "Failed to detect ChArUco board",
                "image_path": str(image_path),
            }

        pixels_per_mm, calibration_success = self.calculate_pixel_to_mm_ratio(
            charuco_corners, charuco_ids
        )

        if not calibration_success:
            return {
                "success": False,
                "error": "Failed to calculate pixel-to-mm ratio",
                "image_path": str(image_path),
            }

        # Detect windows with optimal analysis
        detections, inference_time, loaded_image = (
            self.detect_windows_with_optimal_accuracy(image_path)
        )

        if not detections:
            return {
                "success": True,
                "warning": "No windows detected",
                "pixels_per_mm": pixels_per_mm,
                "windows": [],
            }

        # Process each detection
        results = {
            "success": True,
            "image_path": str(image_path),
            "pixels_per_mm": pixels_per_mm,
            "measurement_method": "optimal_multi_method_analysis",
            "windows": [],
        }

        for detection in detections:
            measurement = self.get_best_measurement(
                detection["mask_analysis"], pixels_per_mm
            )

            if measurement:
                window_result = {
                    "id": detection["id"],
                    "detection_confidence": detection["confidence"],
                    **measurement,
                    "quality_metrics": detection["mask_analysis"]["quality_metrics"],
                }
                results["windows"].append(window_result)

                print(f"Window {detection['id']}:")
                print(f"  Detection Confidence: {detection['confidence']:.3f}")
                print(f"  Measurement Confidence: {measurement['confidence']:.3f}")
                print(f"  Width: {measurement['width_mm']:.1f} mm")
                print(f"  Height: {measurement['height_mm']:.1f} mm")
                print(f"  Methods Used: {measurement['valid_methods']}")

        if output_dir:
            self.save_annotated_image_with_masks(
                loaded_image, results, detections, output_dir
            )

        return results


# Convenience function
def measure_with_optimal_accuracy(image_path, output_dir=None):
    """
    Use the most accurate measurement system available.
    """
    system = OptimalMeasurementSystem()
    return system.process_image_with_optimal_accuracy(image_path, output_dir)


if __name__ == "__main__":
    start_time = time.time()
    output_dir = "measurement_results"
    images_paths = Path(settings.SUPER_TESTER_PATH).glob("*.*")

    # Create results file
    results_file = Path(output_dir) / f"mask_measurement_results_it_is_me_MARIO!.json"
    all_results = []

    for image_path in images_paths:
        print("\n\n", "=" * 40, "\n\n")
        print(" === STARTING NEW WINDOW ===\n")
        try:
            # Use the enhanced mask-based measurement
            # results = measure_window_with_masks(image_path, output_dir)
            results = measure_with_optimal_accuracy(image_path, output_dir)
            all_results.append(results)

            if results["success"]:
                print(f"\n=== MASK-BASED MEASUREMENT RESULTS ===")
                print(f"Image: {results['image_path']}")
                print(f"Calibration: {results['pixels_per_mm']:.3f} pixels/mm")
                print(f"Windows detected: {len(results['windows'])}")

                for window in results["windows"]:
                    print(f"\nWindow {window['id']}:")
                    print(
                        f"  Dimensions: {window['width_mm']:.1f} x {window['height_mm']:.1f} mm"
                    )
                    # print(f"  Detection Confidence: {window['confidence']:.3f}")
                    # print(
                    #     f"  Measurement Confidence: {window['measurement_confidence']:.3f}"
                    # )
                    # print(f"  Mask Area: {window['mask_area_pixels']} pixels")
                    print(f"  Detection Confidence: {window['detection_confidence']:.3f}")
                    print(f"  Measurement Confidence: {window['confidence']:.3f}")
                    print(f"  Mask Area: {window['quality_metrics']['mask_area']} pixels")
            else:
                print(f"Measurement failed: {results.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"Error processing image: {e}")
            all_results.append(
                {"success": False, "error": str(e), "image_path": str(image_path)}
            )
        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print(" \n=== END OF WINDOW ===\n")

    # Write results to file
    os.makedirs(output_dir, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(utils.convert_numpy_types(all_results), f, indent=2)

    print(f"\nResults saved to: {results_file}")
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
