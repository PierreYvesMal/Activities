import cv2
import numpy as np
import os

# =========================
# CONFIG
# =========================
NUM_SECTIONS = 5
DEBUG_DIR = "debug_sections"

os.makedirs(DEBUG_DIR, exist_ok=True)


# =========================
# PREPROCESSING
# =========================
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

    if angle < -45:
        angle = 90 + angle

    return angle


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


# =========================
# PROJECTION PROFILE
# =========================
def compute_profile(bw):
    profile = np.sum(bw, axis=1).astype(np.float32)

    profile /= (np.max(profile) + 1e-6)

    # strong smoothing for scan stability
    kernel = 101
    profile = np.convolve(profile, np.ones(kernel) / kernel, mode="same")

    return profile


# =========================
# CANDIDATE SEPARATORS
# =========================
def find_candidates(profile, window=90, top_k=40):
    """
    Sliding window to find low-density horizontal bands.
    """
    scores = []

    for i in range(len(profile) - window):
        score = np.mean(profile[i:i + window])
        scores.append((score, i + window // 2))

    scores.sort(key=lambda x: x[0])  # lowest density first

    return [pos for _, pos in scores[:top_k]]


# =========================
# ENFORCE EXACTLY K CUTS
# =========================
def enforce_k(candidates, k, min_dist, h):
    selected = []

    for c in sorted(candidates):
        if all(abs(c - s) > min_dist for s in selected):
            selected.append(c)
        if len(selected) == k:
            break

    # fallback: evenly spaced if detection fails
    if len(selected) < k:
        step = h // (k + 1)
        selected = [(i + 1) * step for i in range(k)]

    return sorted(selected)


# =========================
# SPLIT IMAGE
# =========================
def split_image(img, cuts):
    sections = []

    for i in range(len(cuts) - 1):
        y1, y2 = cuts[i], cuts[i + 1]
        sections.append(img[y1:y2, :])

    return sections


# =========================
# DEBUG OUTPUTS
# =========================
def save_debug(img, bw, profile, cuts):
    # overlay
    overlay = img.copy()

    for y in cuts:
        cv2.line(overlay, (0, y), (overlay.shape[1], y), (0, 0, 255), 2)

    cv2.imwrite(f"{DEBUG_DIR}/overlay.png", overlay)

    # sections preview
    for i in range(len(cuts) - 1):
        sec = img[cuts[i]:cuts[i + 1], :]
        cv2.imwrite(f"{DEBUG_DIR}/section_{i}.png", sec)

    # profile visualization (no matplotlib)
    h = 300
    w = len(profile)

    vis = np.zeros((h, w, 3), dtype=np.uint8)

    for x in range(w):
        y = int((1 - profile[x]) * (h - 1))
        cv2.line(vis, (x, h - 1), (x, y), (0, 255, 0), 1)

    cv2.imwrite(f"{DEBUG_DIR}/profile.png", vis)


# =========================
# MAIN PIPELINE
# =========================
def process(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    # 1. initial binarization
    bw = binarize(img)

    # 2. deskew
    img, angle = deskew(img, bw)
    print(f"[INFO] Skew correction: {angle:.2f}°")

    # 3. re-binarize after rotation
    bw = binarize(img)

    # cleanup noise
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 4. projection
    profile = compute_profile(bw)
    h = bw.shape[0]

    # 5. find candidates
    candidates = find_candidates(profile)

    # 6. enforce EXACTLY 4 cuts
    cuts = enforce_k(
        candidates,
        k=NUM_SECTIONS - 1,
        min_dist=h // 12,
        h=h
    )

    cuts = [0] + cuts + [h]

    print("[INFO] Cuts:", cuts)

    # 7. split
    sections = split_image(img, cuts)

    # 8. debug output
    save_debug(img, bw, profile, cuts)

    return sections, cuts


# =========================
# RUN
# =========================
if __name__ == "__main__":
    sections, cuts = process("input.png")
    print(f"[DONE] Extracted {len(sections)} sections")