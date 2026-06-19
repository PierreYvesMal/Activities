import cv2
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import os

def pdf_to_cv_image(pdf_path: str, dpi: int = 400):
    page = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)[0]
    return cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

def crop_schedule_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert so lines become white
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    h, w = bw.shape

    # Extract horizontal lines
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(w // 20, 20), 1)
    )
    horizontal = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # Extract vertical lines
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(h // 20, 20))
    )
    vertical = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    # Combine grid lines
    grid = cv2.bitwise_or(horizontal, vertical)

    # Thicken slightly to connect gaps
    grid = cv2.dilate(
        grid,
        np.ones((5, 5), np.uint8),
        iterations=2
    )

    contours, _ = cv2.findContours(
        grid,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("No grid found")

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    cropped = img[y:y+h, x:x+w]

    return cropped, (x, y, w, h), grid

def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )
    return bw


# =========================
# DESKEW (IMPORTANT)
# =========================
def estimate_skew(bw):
    coords = np.column_stack(np.where(bw > 0))

    if len(coords) < 1000:
        return 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # OpenCV angle normalization
    if angle < -45:
        angle = 90 + angle

    # IMPORTANT FIX:
    # we want to rotate BACK to horizontal
    return -angle


def deskew(img, bw):
    angle = estimate_skew(bw)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated_img, angle

def deskew_image(img):
    bw = binarize(img)
    deskewed, angle = deskew(img, bw)
    print(f"Estimated skew angle: {angle:.2f} degrees")
    return deskewed

import cv2
import numpy as np

def detect_column_separators(img, debug=False):
    """
    Detect vertical column separators in a scanned grid image (assumed 3 columns).

    Args:
        img: input image (BGR or grayscale OpenCV image)
        debug: if True, returns intermediate visualization data

    Returns:
        list of x coordinates of column separators (length = 2)
    """

    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape

    # 2. Binarize (invert so lines/text become white)
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 15
    )

    # 3. Extract vertical structures (grid separators)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 40)))
    vertical_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # 4. Strengthen vertical lines
    vertical_lines = cv2.dilate(vertical_lines, np.ones((3, 3), np.uint8), iterations=2)

    # 5. Vertical projection profile (sum over rows)
    profile = np.sum(vertical_lines > 0, axis=0).astype(np.float32)

    # 6. Smooth profile to reduce noise
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (1, 51), 0).flatten()

    # 7. Suppress margins (avoid picking borders)
    margin = int(0.05 * w)
    profile[:margin] = 0
    profile[-margin:] = 0

    # 8. Find peaks (we expect 2 separators for 3 columns)
    peaks = []
    temp = profile.copy()

    for _ in range(2):
        x = int(np.argmax(temp))
        peaks.append(x)

        # suppress neighborhood to avoid double detection
        left = max(0, x - 30)
        right = min(w, x + 30)
        temp[left:right] = 0

    peaks.sort()

    if debug:
        return peaks, profile, vertical_lines

    return peaks

def save_column_separator_overlay(img, output_path="overlay.png"):
    """
    Detect column separators and save an overlay PNG using OpenCV.

    Args:
        img: input image (grayscale or BGR)
        output_path: where to save the result PNG

    Returns:
        peaks: list of detected x positions
        output_path: saved file path
    """

    # --- detect peaks ---
    peaks = detect_column_separators(img)

    # --- ensure BGR for drawing ---
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    h, _ = vis.shape[:2]

    # --- draw separators ---
    for x in peaks:
        cv2.line(vis, (x, 0), (x, h), (0, 0, 255), 3)
        # cv2.circle(vis, (x, 30), 8, (0, 255, 0), -1)

    return vis, peaks

def split_into_3_sections(img, peaks, out_dir="debug_sections"):
    """
    img: cv2 image (H, W, C)
    peaks: list/array of 2 x-coordinates [x1, x2] (sorted)
    out_dir: output folder
    """

    os.makedirs(out_dir, exist_ok=True)

    h, w = img.shape[:2]

    # ensure sorted and valid
    peaks = sorted(peaks)
    if len(peaks) != 2:
        raise ValueError(f"Expected 2 peaks for 3 columns, got {len(peaks)}")

    x1, x2 = peaks

    # safety clamp
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))

    # split
    sections = [
        img[:, 0:x1],
        img[:, x1:x2],
        img[:, x2:w],
    ]

    # save
    for i, sec in enumerate(sections, start=1):
        path = os.path.join(out_dir, f"v{i}.png")
        cv2.imwrite(path, sec)

    return sections

def find_horizontal_separators(img):
    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape

    # 2. Binarize (invert so lines/text become white)
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 15
    )

    # 3. Extract horizontal structures (grid separators)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, w // 40), 1)
    )
    horizontal_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # 4. Strengthen horizontal lines
    horizontal_lines = cv2.dilate(horizontal_lines, np.ones((3, 3), np.uint8), iterations=2)

    # 5. Horizontal projection profile (sum over columns)
    profile = np.sum(horizontal_lines > 0, axis=1).astype(np.float32)

    # 6. Smooth profile to reduce noise
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (51, 1), 0).flatten()

    # 7. Suppress margins (avoid picking borders)
    margin = int(0.05 * h)
    profile[:margin] = 0
    profile[-margin:] = 0

    # 8. Find peaks (we expect separators as low-signal valleys OR strong lines depending scan)
    peaks = []
    temp = profile.copy()

    for _ in range(5):  # adjust if you expect more separators
        y = int(np.argmax(temp))
        peaks.append(y)

        # suppress neighborhood
        top = max(0, y - 30)
        bottom = min(h, y + 30)
        temp[top:bottom] = 0

    peaks.sort()

    return peaks

def draw_separator_overlay(img, peaks, color=(0, 0, 255), thickness=2):
    """
    Draw horizontal separator lines on the image.

    Args:
        img: input BGR image
        peaks: list of y-coordinates
        color: line color (BGR)
        thickness: line thickness

    Returns:
        overlay image (BGR)
    """

    overlay = img.copy()
    h, w = overlay.shape[:2]

    for y in peaks:
        cv2.line(
            overlay,
            (0, int(y)),
            (w, int(y)),
            color,
            thickness
        )

    return overlay

img = pdf_to_cv_image("schedule.pdf")
img_cropped = crop_schedule_grid(img)[0]
img_deskewed = deskew_image(img_cropped)
img_overlay, peaks = save_column_separator_overlay(img_deskewed)
imgs = split_into_3_sections(img_deskewed, peaks)
horizontal_separators = find_horizontal_separators(img_deskewed)
img_overlay = draw_separator_overlay(img_overlay, horizontal_separators)
cv2.imwrite("final_overlay.png", img_overlay)
# cv2.imshow("cropped", img_cropped)
# cv2.imshow("deskewed", img_deskewed)
# for img in imgs:
#     cv2.imshow("section", img)
#     cv2.waitKey(0)
cv2.imshow("overlay", img_overlay)
cv2.waitKey(0)