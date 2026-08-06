import cv2
import numpy as np
import os
from pdf2image import convert_from_path
from datetime import datetime
import locale
from drive import GoogleDriveClient
import signal
import sys
import time

HEIGHT = 720
WIDTH = 1280

running = True

def signal_handler(sig, frame):
    global running
    print("Shutdown signal received")
    running = False
    cv2.destroyAllWindows()

INPUT_FOLDER = "1rYLC7Dqrb5CpzmLpmdvTkjCiiHAk2Uff"
PROCESSED_FOLDER = "1CDgW498BhuhpJ1uX1F2an7sGsRtxllB1"
OVERLAY_FOLDER = "15D2aJsFI7FKgRF6H62u-q7Tni5b7ppvv"

drive = GoogleDriveClient()

file = drive.find_file_pattern(INPUT_FOLDER, "*.pdf")

if file is None:
    print("Not found")
else:
    drive.download_file(file["id"], "schedule.pdf")
    drive.move_file(file["id"], PROCESSED_FOLDER)
    print("Probably Downloaded and moved schedule.pdf")

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
# CLOCK OVERLAY
# =========================

def create_clock_overlay():
    overlay = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)

    now = datetime.now()
    day = f"{now.strftime('%A').capitalize()} {now.day}"
    clock = now.strftime("%H:%M")

    font = cv2.FONT_HERSHEY_SIMPLEX
    day_scale = 1.5
    clock_scale = 3.0
    thickness = 4
    gap = 25

    (day_w, day_h), day_base = cv2.getTextSize(day, font, day_scale, thickness)
    (clock_w, clock_h), clock_base = cv2.getTextSize(clock, font, clock_scale, thickness)

    padding = 20

    box_w = day_w + gap + clock_w + 2 * padding
    box_h = max(day_h, clock_h) + max(day_base, clock_base) + 2 * padding

    x = WIDTH - box_w - 30
    y = 30

    cv2.rectangle(
        overlay,
        (x, y),
        (x + box_w, y + box_h),
        (0, 0, 0, 160),
        -1
    )

    # Align both texts on their baseline
    baseline_y = y + padding + max(day_h, clock_h)

    cv2.putText(
        overlay,
        day,
        (x + padding, baseline_y),
        font,
        day_scale,
        (255, 255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        overlay,
        clock,
        (x + padding + day_w + gap, baseline_y),
        font,
        clock_scale,
        (255, 255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return overlay

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
drive.push_file(OVERLAY_FOLDER, "final_overlay.png")

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
    # cv2.imshow("Image", today_display)

    # Dimensions of your displayed image
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720

    while True:
        # Your existing computation
        today_display = img[
            h_peaks[day+1]:h_peaks[day+2],
            v_peaks[1]:v_peaks[-1]
        ]

        h, w = today_display.shape[:2]

        # Compute scale while preserving aspect ratio
        scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)

        new_width = int(w * scale)
        new_height = int(h * scale)

        # Resize while keeping aspect ratio
        resized = cv2.resize(
            today_display,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        # Create black 720p background
        background = np.zeros(
            (TARGET_HEIGHT, TARGET_WIDTH, 3),
            dtype=np.uint8
        )

        # Center the image
        x_offset = (TARGET_WIDTH - new_width) // 2
        y_offset = (TARGET_HEIGHT - new_height) // 2

        background[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = resized

        # Blend overlay
        overlay = create_clock_overlay()
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0

        result = (
            background.astype(np.float32) * (1.0 - alpha)
            +
            overlay[:, :, :3].astype(np.float32) * alpha
        ).astype(np.uint8)


        cv2.imshow("Image", result)

        if cv2.waitKey(10000) == 27:
            break

    cv2.destroyAllWindows()