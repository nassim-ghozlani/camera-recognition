"""
Live hand finger counter (MediaPipe Tasks API).

Counts raised fingers on each hand you show the camera. Show two hands and
it counts each separately and displays the sum.

Uses the modern MediaPipe HandLandmarker (Tasks API) because the old
mp.solutions.hands API was removed from recent MediaPipe versions.
The hand-landmark model is downloaded automatically on first run.

How counting works:
  MediaPipe gives 21 landmarks per hand. A finger is "up" when its tip is
  farther from the palm than its middle joint:
    - index/middle/ring/pinky: tip is ABOVE the PIP joint (smaller y)
    - thumb: tip is to the side of its IP joint (x compared, per handedness)

The frame is mirrored (selfie view) so left/right match what you see.

Keys:  q = quit   |   s = save a snapshot

Requires:  pip install mediapipe opencv-python
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CAM_INDEX = 0

# Model is fetched once and cached next to this script.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

# Landmark indices
TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky fingertips
PIPS = [6, 10, 14, 18]   # their PIP joints
THUMB_TIP, THUMB_IP = 4, 3

# Connections for drawing the hand skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand-landmark model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def _dist(a, b):
    """2D distance between two landmarks."""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def count_fingers(landmarks):
    """Return how many fingers are up for one hand (0-5).

    Distance-based so it works at any hand rotation:
      - A finger is extended when its TIP is farther from the wrist (0)
        than its PIP joint.
      - The thumb is extended when its TIP is farther from the pinky
        base (17) than the thumb MCP (2).
    """
    wrist = landmarks[0]
    count = 0

    # Four fingers.
    for tip, pip in zip(TIPS, PIPS):
        if _dist(landmarks[tip], wrist) > _dist(landmarks[pip], wrist):
            count += 1

    # Thumb (rotation-independent, no handedness needed).
    pinky_mcp = landmarks[17]
    if _dist(landmarks[THUMB_TIP], pinky_mcp) > _dist(landmarks[2], pinky_mcp):
        count += 1

    return count


def draw_hand(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def main():
    ensure_model()

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAM_INDEX}. "
            "Try a different CAM_INDEX or check camera permissions."
        )

    snapshot_count = 0
    frame_i = 0
    print("Running. Press 'q' to quit, 's' to save a snapshot.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror for a natural selfie view.
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs a monotonically increasing timestamp (ms).
        timestamp_ms = int(frame_i * 33)
        result = detector.detect_for_video(mp_image, timestamp_ms)
        frame_i += 1

        total = 0
        if result.hand_landmarks:
            for landmarks, handedness in zip(
                result.hand_landmarks, result.handedness
            ):
                label = handedness[0].category_name  # 'Left' / 'Right'
                fingers = count_fingers(landmarks)
                total += fingers

                draw_hand(frame, landmarks, w, h)

                wx = int(landmarks[0].x * w)
                wy = int(landmarks[0].y * h)
                cv2.putText(
                    frame, f"{label}: {fingers}", (wx - 40, wy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

        cv2.rectangle(frame, (0, 0), (260, 60), (0, 0, 0), -1)
        cv2.putText(
            frame, f"Total: {total}", (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3
        )

        cv2.imshow("Hand Finger Counter  (q=quit, s=snapshot)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"hand_snapshot_{snapshot_count:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
            snapshot_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
