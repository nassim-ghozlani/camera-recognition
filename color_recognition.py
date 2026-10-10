"""
Live object color recognition (contour-based).

Pipeline per frame:
  1. Blur + convert BGR -> HSV.
  2. Threshold on saturation/value to separate colourful objects from a
     plain background.
  3. Find external contours; keep ones above a minimum area.
  4. For each contour, sample the median HSV inside its mask and map the
     hue to a named colour.
  5. Draw the box + colour label.

Works best with distinct objects on a plain, evenly-lit background.

Keys:  q = quit   |   s = save a snapshot

Requires:  pip install opencv-python numpy
"""

import cv2
import numpy as np

# ----------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------
CAM_INDEX = 0          # change if you have multiple cameras
MIN_AREA = 1500        # ignore contours smaller than this (pixels^2)
SAT_MIN = 60           # min saturation to count a pixel as "colourful"
VAL_MIN = 40           # min brightness to avoid dark noise
BLUR_KSIZE = 7         # gaussian blur kernel (odd number)

# Hue is 0-179 in OpenCV. Each entry: (name, hue_low, hue_high, BGR_for_label)
# Red wraps around 0/179 so it appears twice.
HUE_RANGES = [
    ("Red",     0,   10, (0,   0, 255)),
    ("Orange",  11,  22, (0, 140, 255)),
    ("Yellow",  23,  33, (0, 255, 255)),
    ("Green",   34,  85, (0, 255,   0)),
    ("Cyan",    86, 100, (255, 255, 0)),
    ("Blue",   101, 130, (255,   0, 0)),
    ("Purple", 131, 160, (255,   0, 255)),
    ("Red",    161, 179, (0,   0, 255)),
]


def classify_color(h, s, v):
    """Map a median HSV sample to a colour name + a BGR colour for drawing."""
    # Low-saturation pixels are greys/whites/blacks regardless of hue.
    if s < SAT_MIN:
        if v < 50:
            return "Black", (40, 40, 40)
        if v > 200:
            return "White", (255, 255, 255)
        return "Gray", (150, 150, 150)

    for name, lo, hi, bgr in HUE_RANGES:
        if lo <= h <= hi:
            return name, bgr
    return "Unknown", (200, 200, 200)


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAM_INDEX}. "
            "Try a different CAM_INDEX or check camera permissions."
        )

    snapshot_count = 0
    print("Running. Press 'q' to quit, 's' to save a snapshot.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        blurred = cv2.GaussianBlur(frame, (BLUR_KSIZE, BLUR_KSIZE), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Mask: keep pixels that are colourful and bright enough.
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = ((sat > SAT_MIN) & (val > VAL_MIN)).astype(np.uint8) * 255

        # Clean up the mask so contours are solid blobs.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Sample only the pixels belonging to THIS contour.
            single = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(single, [cnt], -1, 255, -1)
            pixels = hsv[single == 255]
            if pixels.size == 0:
                continue

            med_h, med_s, med_v = np.median(pixels, axis=0)
            name, bgr = classify_color(int(med_h), int(med_s), int(med_v))

            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
            label = f"{name}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), bgr, -1)
            # Pick readable text colour based on label background brightness.
            txt_col = (0, 0, 0) if sum(bgr) > 400 else (255, 255, 255)
            cv2.putText(
                frame, label, (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_col, 2
            )

        cv2.imshow("Color Recognition  (q=quit, s=snapshot)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"snapshot_{snapshot_count:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
            snapshot_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
