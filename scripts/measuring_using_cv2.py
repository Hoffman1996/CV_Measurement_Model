# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def resize_image_to_fit_screen(image, max_width=1920, max_height=1080):
    """
    Resize image to fit within screen dimensions while maintaining aspect ratio
    """
    height, width = image.shape[:2]

    # Calculate scaling factor to fit within screen bounds
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height, 1.0)  # Don't upscale

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_image(image):
    """
    Enhanced preprocessing for complex images
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Apply adaptive threshold for better edge detection in varying lighting
    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Use morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return gray, thresh


def filter_contours(contours, image_area, min_area_ratio=0.001, max_area_ratio=0.5):
    """
    Filter contours based on area, aspect ratio, and rectangularity
    """
    filtered = []

    # Print some stats for debugging
    areas = [cv2.contourArea(cnt) for cnt in contours]
    if areas:
        print(
            f"Contour areas - Min: {min(areas):.0f}, Max: {max(areas):.0f}, Avg: {np.mean(areas):.0f}"
        )
        print(
            f"Area thresholds - Min: {image_area * min_area_ratio:.0f}, Max: {image_area * max_area_ratio:.0f}"
        )

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        # Filter by area (remove very small and very large contours)
        if area < image_area * min_area_ratio:
            print(
                f"  Contour {i}: area {area:.0f} too small (< {image_area * min_area_ratio:.0f})"
            )
            continue
        if area > image_area * max_area_ratio:
            print(
                f"  Contour {i}: area {area:.0f} too large (> {image_area * max_area_ratio:.0f})"
            )
            continue

        # Check if contour is roughly rectangular
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)

        # Rectangularity: how close the contour is to its bounding rectangle
        rectangularity = area / box_area if box_area > 0 else 0

        if rectangularity < 0.5:  # Relaxed from 0.7 to 0.5
            print(f"  Contour {i}: rectangularity {rectangularity:.2f} too low (< 0.5)")
            continue

        # Check aspect ratio (avoid very thin objects)
        width, height = rect[1]
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 15:  # Relaxed from 10 to 15
                print(f"  Contour {i}: aspect ratio {aspect_ratio:.1f} too high (> 15)")
                continue

        print(
            f"  Contour {i}: PASSED - area={area:.0f}, rect={rectangularity:.2f}, ratio={aspect_ratio:.1f}"
        )
        filtered.append(cnt)

    return filtered


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument(
    "-w",
    "--width",
    type=float,
    required=True,
    help="width of the left-most object in the image (in inches)",
)
ap.add_argument(
    "--min-area",
    type=float,
    default=0.00001,
    help="minimum contour area as ratio of image area (default: 0.00001)",
)
ap.add_argument(
    "--max-area",
    type=float,
    default=0.8,
    help="maximum contour area as ratio of image area (default: 0.8)",
)
ap.add_argument(
    "--debug", action="store_true", help="show debug images for preprocessing steps"
)
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
if image is None:
    print("Error: Could not load image")
    exit()

print(f"Image size: {image.shape[1]}x{image.shape[0]}")
image_area = image.shape[0] * image.shape[1]

# Enhanced preprocessing
gray, processed = preprocess_image(image)

if args["debug"]:
    debug_gray, _ = resize_image_to_fit_screen(gray)
    debug_processed, _ = resize_image_to_fit_screen(processed)
    cv2.imshow("Grayscale", debug_gray)
    cv2.imshow("Processed", debug_processed)
    cv2.waitKey(0)
    cv2.destroyWindow("Grayscale")
    cv2.destroyWindow("Processed")

# Find contours using the processed image
cnts = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

print(f"Found {len(cnts)} initial contours")

# Filter contours
cnts = filter_contours(cnts, image_area, args["min_area"], args["max_area"])
print(f"After filtering: {len(cnts)} contours")

if len(cnts) == 0:
    print(
        "No suitable contours found. Try adjusting --min-area and --max-area parameters"
    )
    exit()

# Sort the contours from left-to-right
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# Create a copy for drawing all detections
all_detections = image.copy()

# loop over the contours individually
for i, c in enumerate(cnts):
    print(f"\nProcessing contour {i+1}/{len(cnts)}")
    print(f"Contour area: {cv2.contourArea(c):.0f} pixels")

    # compute the rotated bounding box of the contour
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left order
    box = perspective.order_points(box)

    # Draw on both individual and combined images
    for img in [orig, all_detections]:
        cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)
        # Draw corner points
        for x, y in box:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoints
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on both images
    for img in [orig, all_detections]:
        cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(
            img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2
        )
        cv2.line(
            img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2
        )

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    print(f"Pixel distances - Height: {dA:.1f}, Width: {dB:.1f}")

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]
        print(
            f"Calibration: {pixelsPerMetric:.2f} pixels per inch (using first object)"
        )

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    print(f"Real dimensions - Height: {dimA:.2f}in, Width: {dimB:.2f}in")

    # draw the object sizes on both images
    for img in [orig, all_detections]:
        cv2.putText(
            img,
            "{:.1f}in".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "{:.1f}in".format(dimB),
            (int(trbrX + 10), int(trbrY)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # Add object number
        cv2.putText(
            img,
            f"#{i+1}",
            (int(tlblX - 25), int(tlblY + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    # Resize and show individual detection
    display_image, scale_factor = resize_image_to_fit_screen(orig, 1920, 1080)
    cv2.imshow(f"Object #{i+1} Detection", display_image)
    print("Press any key to continue to next object, or 'q' to quit...")
    key = cv2.waitKey(0)
    cv2.destroyWindow(f"Object #{i+1} Detection")

    if key == ord("q"):
        break

# Show all detections together
if len(cnts) > 1:
    display_all, scale_factor = resize_image_to_fit_screen(all_detections, 1920, 1080)
    print(
        f"\nDisplay scale: {scale_factor:.2f}x (Image resized to fit 1920x1080 screen)"
    )
    cv2.imshow("All Detections", display_all)
    print("Press any key to exit...")
    cv2.waitKey(0)

cv2.destroyAllWindows()
print(f"\nProcessing complete. Found and measured {len(cnts)} objects.")
