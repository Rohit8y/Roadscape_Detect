import os
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Implementation of road marking detection')
parser.add_argument('--data_path', default='data/road_marking/', type=str,
                    help='the path to the dataset')
parser.add_argument('--output_path', default='output/', type=str,
                    help='output directory for the binary mask')
# ------------------------------Canny Properties------------------------------------------#
parser.add_argument('-canny_lower_threshold', default=40, type=int,
                    help='lower threshold value in Hysteresis Thresholding')
parser.add_argument('-canny_upper_threshold', default=130, type=int,
                    help='upper threshold value in Hysteresis Thresholding')
# ------------------------------Hough Properties------------------------------------------#
parser.add_argument('-hough_threshold', default=30, type=int,
                    help='threshold value to detect lines')
parser.add_argument('-hough_min_length', default=100, type=int,
                    help='minimum length of line')
parser.add_argument('-hough_max_gap', default=10, type=int,
                    help='maximum gap used to combine lines')

args = parser.parse_args()


def preprocess(grayscale_image):
    """
    Apply CLAHE contrast, bilateral and gaussian to the input
    grayscale image

    Args:
        grayscale_image: Input grayscale image

    Returns:
        processed_image:  Preprocessed output image
    """
    # Create a CLAHE object and apply contrast
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
    gray = clahe.apply(grayscale_image)

    # Apply Bilateral and Gaussian Blur
    bilateral = cv2.bilateralFilter(gray, 5, 150, 150)
    processed_image = cv2.GaussianBlur(bilateral, (5, 5), 0)

    return processed_image


def get_roi(input_image):
    """
    Extract the region of interest (ROI) from the input image.
    A polygon mask is defined as the primary ROI. The car mask
    is removed from the polygon mask to extract final ROI.

    Args:
        input_image: Input grayscale image

    Returns:
        roi: Region of Interest

    """
    # Generate blanks masks
    polygon_mask = np.zeros_like(input_image)
    car_mask = np.zeros_like(input_image)

    rows, cols = input_image.shape[:2]

    # Polygon vertices
    bottom_left = [0, rows]
    top_left = [cols * 0.3, rows * 0.52]
    bottom_right = [cols, rows]
    top_right = [cols * 0.65, rows * 0.52]
    poly_vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    # Car vertices
    car_1 = [0, rows]
    car_2 = [cols * 0.19, rows * 0.78]
    car_3 = [cols * 0.48, rows * 0.73]
    car_4 = [cols * 0.75, rows * 0.8]
    car_5 = [cols, rows]
    car_vertices = np.array([[car_1, car_2, car_3, car_4, car_5]], dtype=np.int32)

    # Filling the polygon with white color and generating the road mask
    cv2.fillPoly(polygon_mask, poly_vertices, 255)
    cv2.fillPoly(car_mask, car_vertices, 255)
    road_mask = np.subtract(polygon_mask, car_mask)

    # Bitwise AND on the input image and road mask to get only the edges on the road
    return cv2.bitwise_and(input_image, road_mask)


def apply_dilation_erosion(input, iterations=1):
    """
    Apply dilation and erosion to the input image for n iterations.

    Args:
        input: Input binary mask
        iterations: Number of iterations to run

    Returns:
        output: Output binary image

    """
    kernel = np.ones((3, 3), np.uint8)
    input = cv2.dilate(input, kernel, iterations=iterations)
    output = cv2.erode(input, kernel, iterations=iterations)
    return output


def get_contours(image, min_area=10):
    """
    Extract list of contours from given input image with
    area greater than min_area.

    Args:
        image: Input Image
        min_area: Minimum area of contour

    Returns:
        filtered_contours: List of contours with area greater than min_area


    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered_contours


def compute_mask(image):
    """
    Compute the final mask using Canny edge detection and Hough transform
    from input RGB image.

    Args:
        image: Given input RGB image

    Returns:
        final_mask: The output mask with road markings

    """
    # Read image and convert to grayscale
    img = cv2.imread(args.data_path + image)
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocessing
    input_image = preprocess(grayscale_image)

    # Apply Canny Edge detection
    canny = cv2.Canny(input_image.copy(), args.canny_lower_threshold, args.canny_upper_threshold)

    # Get Region of Interest
    roi = get_roi(canny)
    roi_refined = apply_dilation_erosion(roi)

    # Hough Transform
    houghLines = cv2.HoughLinesP(roi_refined, 1, np.pi / 180, args.hough_threshold, minLineLength=args.hough_min_length,
                                 maxLineGap=args.hough_max_gap)
    hough = np.zeros_like(input_image)
    for line in houghLines:
        x1, y1, x2, y2 = line[0]
        cv2.line(hough, (x1, y1), (x2, y2), (255, 255, 255), 5)

    # Get Contours of hough lines
    lines_mask = np.zeros_like(input_image)
    line_contours_hough = get_contours(hough.copy())
    cv2.drawContours(lines_mask, line_contours_hough, -1, (255), thickness=cv2.FILLED)

    # Get contours of canny roi
    contour_mask = np.zeros_like(input_image)
    filtered_contours_canny = get_contours(roi_refined.copy(), min_area=40)
    cv2.drawContours(contour_mask, filtered_contours_canny, -1, (255), thickness=cv2.FILLED)

    # Combine the mask
    final_mask = np.add(lines_mask, contour_mask)
    return final_mask


def superresolution(mask, scale_factor=2):
    """
    Generate super resolution mask given the binary mask using scaling factor.

    Args:
        mask: Input binary mask
        scale_factor: Scale factor for super resolution

    Returns:
        super_mask: Super resolution mask

    """
    # Upsample the mask using bilinear interpolation
    super_mask = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    return super_mask


if __name__ == '__main__':

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Extract data
    imageList = os.listdir(args.data_path)
    for image in imageList:
        # Compute the binary mask of size 2048 x 2048
        mask = compute_mask(image)
        # Bonus: Increase the resolution of the mask to 4096 x 4096
        final_mask = superresolution(mask)
        cv2.imwrite(os.path.join(args.output_path, image), final_mask)
