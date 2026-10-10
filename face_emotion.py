"""
Live facial emotion recognition (DeepFace).

For each webcam frame we detect faces and classify the dominant emotion
(angry, disgust, fear, happy, sad, surprise, neutral). A box + label is
drawn on every detected face.

Emotion analysis is heavy, so we only re-run it every ANALYZE_EVERY frames
and reuse the last result in between. This keeps the video smooth.

Keys:  q = quit   |   s = save a snapshot

First run downloads the emotion model weights (~a few hundred MB, once).

Requires:  pip install deepface opencv-python tf-keras
"""

import cv2
from deepface import DeepFace

# ----------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------
CAM_INDEX = 0            # change if the wrong camera opens
ANALYZE_EVERY = 5        # run DeepFace every N frames (higher = smoother, laggier labels)
DETECTOR = "opencv"      # face detector backend: 'opencv' (fast) or 'retinaface' (accurate)
SHOW_SCORE = True        # show the confidence % next to the emotion

# A color per emotion for the boxes (BGR).
EMOTION_COLORS = {
    "angry":    (0,   0, 255),
    "disgust":  (0, 128,   0),
    "fear":     (128, 0, 128),
    "happy":    (0, 215, 255),
    "sad":      (255, 128,  0),
    "surprise": (0, 255, 255),
    "neutral":  (200, 200, 200),
}


def analyze(frame):
    """Return DeepFace's list of face results, or [] if none/failure."""
    try:
        # enforce_detection=False -> don't raise when no face is found.
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            detector_backend=DETECTOR,
            enforce_detection=False,
            silent=True,
        )
        # analyze returns a dict for one face or a list for many; normalise.
        if isinstance(results, dict):
            results = [results]
        # Drop phantom detections that cover the whole frame with no real face.
        return [r for r in results if r.get("face_confidence", 1) > 0]
    except Exception as e:
        print("analyze error:", e)
        return []


def draw(frame, results):
    for r in results:
        region = r.get("region", {})
        x, y = region.get("x", 0), region.get("y", 0)
        w, h = region.get("w", 0), region.get("h", 0)
        if w == 0 or h == 0:
            continue

        emotion = r.get("dominant_emotion", "?")
        score = r.get("emotion", {}).get(emotion, 0)
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        label = f"{emotion} {score:.0f}%" if SHOW_SCORE else emotion

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), color, -1)
        txt_col = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(
            frame, label, (x + 2, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_col, 2
        )


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAM_INDEX}. "
            "Try a different CAM_INDEX or check camera permissions."
        )

    print("Loading model (first run downloads weights)... ")
    frame_i = 0
    last_results = []
    snapshot_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Only run the heavy analysis every ANALYZE_EVERY frames.
        if frame_i % ANALYZE_EVERY == 0:
            last_results = analyze(frame)
        frame_i += 1

        draw(frame, last_results)

        cv2.imshow("Emotion Recognition  (q=quit, s=snapshot)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"emotion_snapshot_{snapshot_count:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
            snapshot_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
