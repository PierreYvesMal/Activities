import cv2
import numpy as np
import os
from pdf2image import convert_from_path
from datetime import datetime
import locale
from drive import GoogleDriveClient
import signal
import pytesseract
import re
from enum import Enum, auto

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
PROCESSED_IMAGE_PATH = "processed.npy"
PEAKS_PATH = "peaks.npz"
DAY_START_HOUR = 7
DAY_END_HOUR = 20
NIGHT_START_HOUR = 22
DISPLAY_TIMEOUT_MS = 10000


class State(Enum):
    INIT = auto()
    CHECK_TIME = auto()
    ACTIVITIES = auto()
    FETCH_DRIVE = auto()
    PROCESS_IMAGE = auto()
    VALIDATE_SCHEDULE = auto()
    DISPLAY_PROCESSED = auto()
    DISPLAY_BLACK = auto()
    DISPLAY_ERROR = auto()

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


def save_processed(img, v_peaks, h_peaks):
    np.save(PROCESSED_IMAGE_PATH, img)
    np.savez(PEAKS_PATH, v_peaks=np.array(v_peaks), h_peaks=np.array(h_peaks))


def load_processed():
    img = np.load(PROCESSED_IMAGE_PATH)
    data = np.load(PEAKS_PATH)
    return img, data["v_peaks"].tolist(), data["h_peaks"].tolist()


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
# CANVAS
# =========================

def fit_to_canvas(img, target_width=WIDTH, target_height=HEIGHT):
    """Resize img to fit within target_width x target_height while preserving
    aspect ratio, then center it on a black canvas of the target size."""
    h, w = img.shape[:2]

    # Compute scale while preserving aspect ratio
    scale = min(target_width / w, target_height / h)

    new_width = int(w * scale)
    new_height = int(h * scale)

    # Resize while keeping aspect ratio
    resized = cv2.resize(
        img,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Create black canvas
    canvas = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8
    )

    # Center the image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


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


def blend_overlay(background, overlay):
    """Alpha-composite an RGBA overlay onto a BGR background."""
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    return (
        background.astype(np.float32) * (1.0 - alpha)
        + overlay[:, :, :3].astype(np.float32) * alpha
    ).astype(np.uint8)


# =========================
# OCR
# =========================

def ocr(img):
    """Extract the day name and number from a cropped schedule header cell.

    The input image is expected to be a clean, deskewed crop containing
    only text like 'Vendredi 19'. Returns the recognized text as a string.
    """
    # Convert to grayscale if needed
    gray = to_gray(img)

    # Upscaling and binarizing made the OCR worse.

    # OCR config: uniform block of text (handles 2 lines: "Vendredi" + "19"),
    # French language, alphanumeric whitelist only
    config = (
        '--oem 3 --psm 6 '
        '-c tessedit_char_whitelist='
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
    )
    text = pytesseract.image_to_string(gray, lang='fra', config=config)

    return text.strip()


# =========================
# PIPELINE
# =========================

def classify_time(now):
    if now.weekday() >= 5 or now.hour < DAY_START_HOUR or now.hour >= NIGHT_START_HOUR:
        return State.DISPLAY_BLACK
    if now.hour >= DAY_END_HOUR:
        return State.DISPLAY_ERROR
    return State.ACTIVITIES


def state_without_new_file(schedule_valid, has_processed_image, has_local_pdf):
    if has_processed_image:
        return State.DISPLAY_ERROR if schedule_valid is False else State.DISPLAY_PROCESSED
    if has_local_pdf:
        return State.PROCESS_IMAGE
    return State.DISPLAY_ERROR


def load_or_process_local_schedule():
    if os.path.exists(PROCESSED_IMAGE_PATH) and os.path.exists(PEAKS_PATH):
        return load_processed()

    print("No processed image found, processing local file...")
    img = pdf_to_cv_image("schedule.pdf")
    img = deskew(crop_schedule_grid(img))
    v_peaks, _ = detect_vertical_separators(img)
    h_peaks = detect_horizontal_separators(img)
    save_processed(img, v_peaks, h_peaks)
    return img, v_peaks, h_peaks


def process_new_schedule(drive_client):
    img = pdf_to_cv_image("schedule.pdf")
    img = deskew(crop_schedule_grid(img))
    v_peaks, _ = detect_vertical_separators(img)
    h_peaks = detect_horizontal_separators(img)
    save_processed(img, v_peaks, h_peaks)

    img_v = draw_vertical_overlay(img, v_peaks)
    img_final = draw_horizontal_overlay(img_v, h_peaks)
    cv2.imwrite("final_overlay.png", img_final)
    try:
        drive_client.push_file(OVERLAY_FOLDER, "final_overlay.png")
    except Exception as exc:
        print(f"Could not upload debug overlay: {exc}")

    return img, v_peaks, h_peaks


def schedule_is_up_to_date(img, v_peaks, h_peaks, today):
    day = today.weekday()
    if day + 2 >= len(h_peaks) or len(v_peaks) < 2:
        print("Schedule separators do not contain today's header cell")
        return False

    header = img[h_peaks[day + 1]:h_peaks[day + 2], v_peaks[0]:v_peaks[1]]
    text = ocr(header)
    print(f"OCR: {text}")
    match = re.search(r"\d+", text)
    ocr_day = int(match.group()) if match else None
    if ocr_day == today.day:
        print(f"OCR day ({ocr_day}) matches today ({today.day})")
        return True

    print(f"Schedule is not up to date (OCR day: {ocr_day}, today: {today.day})")
    return False


def make_error_background():
    background = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        background,
        "SCHEDULE ERROR",
        (70, HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    return background


def show_background(background):
    result = blend_overlay(background, create_clock_overlay())
    cv2.imshow("Image", result)
    key = cv2.waitKey(DISPLAY_TIMEOUT_MS)
    if key in (13, 27, ord("q")):
        global running
        running = False


def main():
    global running
    state = State.INIT
    drive_client = None
    img = v_peaks = h_peaks = None
    schedule_valid = None

    while running:
        if state is State.INIT:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            try:
                locale.setlocale(locale.LC_TIME, "fr_BE.UTF-8")
            except locale.Error as exc:
                print(f"French locale unavailable; using system locale: {exc}")
            drive_client = GoogleDriveClient()
            cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                "Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            state = State.CHECK_TIME

        elif state is State.CHECK_TIME:
            state = classify_time(datetime.now())

        elif state is State.ACTIVITIES:
            state = State.FETCH_DRIVE

        elif state is State.FETCH_DRIVE:
            try:
                new_file = drive_client.find_file_pattern(INPUT_FOLDER, "*.pdf")
                if new_file is not None:
                    drive_client.download_file(new_file["id"], "schedule.pdf")
                    drive_client.move_file(new_file["id"], PROCESSED_FOLDER)
                    print("Downloaded and moved a new schedule PDF")
                    schedule_valid = None
                    state = State.PROCESS_IMAGE
                else:
                    has_processed_image = (
                        os.path.exists(PROCESSED_IMAGE_PATH)
                        and os.path.exists(PEAKS_PATH)
                    )
                    has_local_pdf = os.path.exists("schedule.pdf")
                    state = state_without_new_file(
                        schedule_valid, has_processed_image, has_local_pdf
                    )
                    if has_processed_image:
                        img, v_peaks, h_peaks = load_processed()
                    elif not has_local_pdf:
                        print("No new schedule or processed image found")
            except Exception as exc:
                print(f"Drive fetch failed: {exc}")
                state = State.DISPLAY_ERROR

        elif state is State.PROCESS_IMAGE:
            try:
                img, v_peaks, h_peaks = process_new_schedule(drive_client)
                state = State.VALIDATE_SCHEDULE
            except Exception as exc:
                print(f"Schedule processing failed: {exc}")
                schedule_valid = False
                state = State.DISPLAY_ERROR

        elif state is State.VALIDATE_SCHEDULE:
            try:
                schedule_valid = schedule_is_up_to_date(
                    img, v_peaks, h_peaks, datetime.now()
                )
                state = (
                    State.DISPLAY_PROCESSED
                    if schedule_valid
                    else State.DISPLAY_ERROR
                )
            except Exception as exc:
                print(f"Schedule OCR failed: {exc}")
                schedule_valid = False
                state = State.DISPLAY_ERROR

        elif state is State.DISPLAY_PROCESSED:
            try:
                today = datetime.now()
                row = today.weekday()
                if img is None:
                    img, v_peaks, h_peaks = load_or_process_local_schedule()
                if row + 2 >= len(h_peaks) or len(v_peaks) < 2:
                    raise ValueError("Schedule separators do not contain today's row")
                today_display = img[
                    h_peaks[row + 1]:h_peaks[row + 2],
                    v_peaks[1]:v_peaks[-1],
                ]
                show_background(fit_to_canvas(today_display))
            except Exception as exc:
                print(f"Could not display processed schedule: {exc}")
                show_background(make_error_background())
            state = State.CHECK_TIME

        elif state is State.DISPLAY_BLACK:
            show_background(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
            state = State.CHECK_TIME

        elif state is State.DISPLAY_ERROR:
            show_background(make_error_background())
            state = State.CHECK_TIME

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()