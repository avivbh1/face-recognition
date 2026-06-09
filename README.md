# Face Recognition

Real-time multi-face verification system using a webcam (or multiple cameras). Powered by [InsightFace](https://github.com/deepinsight/insightface) (ArcFace), OpenCV, and a Flask web UI.

## Features

- **Multi-camera** — runs all connected cameras in parallel, each with its own live stream
- **Multi-face database** — enroll as many people as you want, each with a name, photo, and custom alert sound
- **Live web UI** — view all camera feeds in your browser with real-time detection badges
- **Recognition alerts** — when a known face is detected, a modal pops up with their photo, name, and which camera saw them, and plays an alert sound
- **Per-camera sounds** — assign a different sound to each camera that plays on any recognition event
- **Per-person sounds** — assign a sound to a specific person that plays when they are recognized

## Requirements

- Python 3.8+
- Webcam(s)

## Installation

```bash
pip install -r requirements.txt
```

> InsightFace will automatically download the `buffalo_sc` model (~30 MB) on first run.

## Running

```bash
python server.py
```

Then open **http://localhost:5000** in your browser.

## How to Use

### Enrolling a Face

1. Open the web UI at `http://localhost:5000`
2. In the **Add Person** panel, type a name
3. Select which camera to use for enrollment
4. Optionally upload an alert sound (MP3, WAV, OGG)
5. Click **Enroll Face** and look at the selected camera
6. Hold still — 5 frames are captured and averaged into a reference embedding

### Recognizing Faces

Recognition runs automatically once faces are enrolled. When a known face appears:
- The camera badge turns green and shows their name
- A modal alert appears with their photo, name, and the camera that detected them
- Their alert sound plays (camera sound takes priority over person sound; falls back to a default beep)

The alert has a 12-second cooldown per person per camera to avoid repeating.

### Camera Settings

- **Rename** — double-click a camera label or click the ✏️ icon
- **Upload sound** — click the 🔊 icon on a camera card to assign a sound that plays for any recognition on that camera

### Removing a Face

Click the **✕** button next to any enrolled face in the **Enrolled Faces** list.

## Tuning

The default similarity threshold is `0.4`. Lower values are stricter (fewer false positives); higher values are more lenient.

To change it, edit `THRESHOLD` at the top of `server.py`:

```python
THRESHOLD = 0.4
```

## File Structure

```
face-recognition/
├── server.py           # Flask server, camera loops, REST API
├── verifier.py         # InsightFace wrapper (embedding + verification)
├── face_db.py          # Face database (save/load/delete enrolled faces)
├── camera_manager.py   # Camera auto-detection and config
├── main.py             # Optional CLI entry point
├── overlay.py          # OpenCV drawing helpers
├── reference.py        # Single-face reference helpers (legacy CLI)
├── requirements.txt
└── static/
    └── index.html      # Web UI
```

Runtime data (git-ignored):

```
faces/              # Enrolled face embeddings, photos, sounds
cameras.json        # Camera labels (auto-created)
static/cam_sounds/  # Per-camera sound files
```
