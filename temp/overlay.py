import time
import cv2
import numpy as np
from multiprocessing import shared_memory

WIDTH = 1280
HEIGHT = 720

shm = shared_memory.SharedMemory(
    create=True,
    size=HEIGHT * WIDTH * 4,
    name="overlay"
)

overlay = np.ndarray(
    (HEIGHT, WIDTH, 4),
    dtype=np.uint8,
    buffer=shm.buf
)

try:
    while True:
        # Clear overlay (fully transparent)
        overlay[:] = 0

        clock = time.strftime("%H:%M:%S")

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 3
        thickness = 5

        (tw, th), baseline = cv2.getTextSize(
            clock,
            font,
            scale,
            thickness
        )

        x = WIDTH - tw - 50
        y = th + 50

        # Transparent black background
        cv2.rectangle(
            overlay,
            (x - 20, y - th - 20),
            (x + tw + 20, y + baseline + 20),
            (0, 0, 0, 160),
            -1
        )

        # White clock
        cv2.putText(
            overlay,
            clock,
            (x, y),
            font,
            scale,
            (255, 255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        time.sleep(0.1)

finally:
    shm.close()
    shm.unlink()