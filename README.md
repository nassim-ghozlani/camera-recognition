# Camera Recognition

Real-time computer-vision demos that run on your webcam with Python and OpenCV:

| Script | What it does | Built with |
|---|---|---|
| `face_emotion.py` | Detects faces and labels each one with its emotion (angry, disgust, fear, happy, sad, surprise, neutral) | [DeepFace](https://github.com/serengil/deepface) |
| `hand_counter.py` | Counts raised fingers on one or two hands and shows the total | [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) |
| `color_recognition.py` | Finds colourful objects and names their colour | OpenCV (HSV + contours) |

## Getting started

Requires **Python 3.9–3.12** and a webcam.

```bash
git clone https://github.com/<your-username>/camera-recognition.git
cd camera-recognition

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Then run any demo:

```bash
python face_emotion.py
python hand_counter.py
python color_recognition.py
```

A window opens showing your camera. In every demo, press **`q`** to quit and **`s`** to save a snapshot.

> **Model files are downloaded automatically on first run.** DeepFace fetches its emotion weights (a few hundred MB, one time) and `hand_counter.py` downloads `hand_landmarker.task` (~8 MB) next to the script. They are not stored in the repo.

## How it works

### Facial emotion (`face_emotion.py`)
1. **Detect faces** using the chosen DeepFace backend (`opencv` = fast, `retinaface` = more accurate).
2. **Classify** each face with DeepFace's emotion CNN, which scores all seven emotions; the highest is the dominant one.
3. **Run sparingly.** The model only re-runs every `ANALYZE_EVERY` frames and reuses the last result in between, so the video stays smooth.
4. **Draw** a colour-coded box and label (with confidence %) per face.

### Finger counter (`hand_counter.py`)
MediaPipe returns 21 landmarks per hand. A finger counts as raised when its tip is farther from the wrist than its middle (PIP) joint; the thumb is compared against the pinky base. These distance checks work at any hand rotation. The view is mirrored so left and right match what you see.

### Colour recognition (`color_recognition.py`)
The frame is blurred and converted to HSV, then saturated, bright pixels are masked and cleaned into solid blobs. For each blob above `MIN_AREA`, the median hue is mapped to a colour name. Works best with distinct objects on a plain, evenly lit background.

## Configuration

Each script has a block of settings at the top:

| Setting | Script | Purpose |
|---|---|---|
| `CAM_INDEX` | all | Which camera to open (try `1`, `2`… if the wrong one opens) |
| `ANALYZE_EVERY` | emotion | How often the model runs: lower is snappier, higher is smoother |
| `DETECTOR` | emotion | `opencv` (fast) or `retinaface` (better in hard lighting or angles) |
| `SHOW_SCORE` | emotion | Show the confidence % |
| `MIN_AREA`, `SAT_MIN`, `VAL_MIN` | color | Object size and colour thresholds |

## Troubleshooting

- **Camera won't open:** change `CAM_INDEX` or check your OS camera permissions.
- **First run is slow:** model weights are downloading; later runs start quickly.
- **No face box:** improve lighting, face the camera, or try `DETECTOR = "retinaface"`.
- **Emotion label flickers:** expressions are subtle; raise `ANALYZE_EVERY` slightly.

## Project structure

```
.
├── color_recognition.py   # colour detection demo
├── face_emotion.py        # emotion recognition demo
├── hand_counter.py        # finger counter demo
├── requirements.txt
├── LICENSE
└── README.md
```

## Ideas for next steps

- Smooth the emotion label across frames
- Log emotions over time to CSV and chart them
- Add age and gender (DeepFace `actions`)
- Recognise hand gestures, not just finger counts

## License

[MIT](LICENSE)
