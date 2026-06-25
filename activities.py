import cv2
import numpy as np
import os
from pdf2image import convert_from_path
from datetime import datetime
import locale

# TODO
# Don't re-process everything. Check if converted image exists. Clean converted when new schedule.pdf arrives.

# =========================
# IO / BASIC UTILITIES
# =========================

def pdf_to_cv_image(pdf_path: str, dpi: int = 400):
    page = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)[0]
    return cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()


def binarize(img, block_size=31, C=15, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C):
    gray = to_gray(img)
    return cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY_INV, block_size, C)


def suppress_margins(profile, margin_ratio):
    n = len(profile)
    m = int(n * margin_ratio)
    profile[:m] = 0
    profile[-m:] = 0
    return profile


def remove_neighborhood_1d(arr, center, radius):
    left = max(0, center - radius)
    right = min(len(arr), center + radius)
    arr[left:right] = 0


# =========================
# CROPPING
# =========================

def crop_schedule_grid(img):
    bw = binarize(img, 31, 15)
    h, w = bw.shape

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 20, 20), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 20, 20)))

    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)

    grid = cv2.bitwise_or(horizontal, vertical)
    grid = cv2.dilate(grid, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No grid found")

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y:y + h, x:x + w]


# =========================
# DESKEW
# =========================

def estimate_skew(bw):
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 1000:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90

    return -angle


def deskew(img):
    bw = binarize(img)
    angle = estimate_skew(bw)

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# =========================
# SEPARATOR DETECTION
# =========================

def detect_vertical_separators(img):
    gray = to_gray(img)
    h, w = gray.shape

    bw = binarize(gray, 35, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 40)))

    v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    v = cv2.dilate(v, np.ones((3, 3), np.uint8), iterations=2)

    profile = np.sum(v > 0, axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (1, 51), 0).flatten()
    # profile = suppress_margins(profile, 0.05)

    peaks = []
    temp = profile.copy()

    for _ in range(4):
        x = int(np.argmax(temp))
        peaks.append(x)
        remove_neighborhood_1d(temp, x, 30)

    return sorted(peaks), v


def detect_horizontal_separators(img):
    gray = to_gray(img)
    h, w = gray.shape

    bw = binarize(gray, 35, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 40), 1))

    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    h_lines = cv2.dilate(h_lines, np.ones((3, 3), np.uint8), iterations=2)

    profile = np.sum(h_lines > 0, axis=1).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (51, 1), 0).flatten()
    # profile = suppress_margins(profile, 0.05)

    peaks = []
    temp = profile.copy()

    for _ in range(7):
        y = int(np.argmax(temp))
        peaks.append(y)
        remove_neighborhood_1d(temp, y, 30)

    return sorted(peaks)


# =========================
# VISUALIZATION
# =========================

def draw_vertical_overlay(img, peaks):
    out = img.copy()
    h = out.shape[0]

    for x in peaks:
        cv2.line(out, (x, 0), (x, h), (0, 0, 255), 2)

    return out


def draw_horizontal_overlay(img, peaks):
    out = img.copy()
    h, w = out.shape[:2]

    for y in peaks:
        cv2.line(out, (0, y), (w, y), (0, 0, 255), 2)

    return out

# =========================
# SPLITTING
# =========================

def split_into_3(img, peaks, out_dir="debug_sections"):
    os.makedirs(out_dir, exist_ok=True)

    x1, x2 = sorted(peaks)
    h, w = img.shape[:2]

    sections = [
        img[:, :x1],
        img[:, x1:x2],
        img[:, x2:]
    ]

    for i, sec in enumerate(sections, 1):
        cv2.imwrite(os.path.join(out_dir, f"v{i}.png"), sec)

    return sections

# =========================
# PIPELINE
# =========================

img = pdf_to_cv_image("schedule.pdf")

img = crop_schedule_grid(img)
img = deskew(img)

v_peaks, _ = detect_vertical_separators(img)
img_v = draw_vertical_overlay(img, v_peaks)

h_peaks = detect_horizontal_separators(img)
img_final = draw_horizontal_overlay(img_v, h_peaks)

cv2.imwrite("final_overlay.png", img_final)

locale.setlocale(locale.LC_TIME, "fr_BE.UTF-8")  # or fr_FR.UTF-8

today = datetime.now()
print(today.strftime("%A %d %B %Y"))
day=today.weekday()
# print("Vertical peaks (columns):", v_peaks)
# print("Horizontal peaks (rows):", h_peaks)

cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(
    "Image",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

if day >= len(h_peaks) - 2:
    print("No schedule available for today.")
else:
    today_display = img[h_peaks[day+1]:h_peaks[day+2], v_peaks[0]:v_peaks[-1]]
    cv2.imwrite("today_section.png", today_display)
    cv2.imshow("Image", today_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
