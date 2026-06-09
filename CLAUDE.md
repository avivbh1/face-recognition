# Face Verification Project

## What this is
Real-time 1:1 face verification from a camera stream.
Stack: Python, InsightFace (ArcFace), OpenCV, ONNX Runtime (CPU only).

## File structure
- main.py        — CLI entry point + stream loop
- verifier.py    — InsightFace wrapper (FaceVerifier class)
- reference.py   — save/load reference embedding (reference.npy)
- overlay.py     — all OpenCV drawing utilities

## Key decisions
- buffalo_sc model (lightweight, good CPU perf)
- Recognition runs every 5 frames to reduce CPU load
- Enrollment averages 5 samples for a more robust reference
- Threshold default 0.4, tunable via --threshold flag

## Run
pip install -r requirements.txt
python main.py --enroll
python main.py
