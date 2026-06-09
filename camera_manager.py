import json
import cv2
from pathlib import Path

CONFIG_FILE = Path("cameras.json")
CAM_SOUNDS_DIR = Path("static") / "cam_sounds"


def detect_cameras(max_check=4):
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available or [0]


def load_config():
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE) as f:
        return {int(k): v for k, v in json.load(f).items()}


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump({str(k): v for k, v in config.items()}, f, indent=2)


def get_cam_sound_path(cam_id):
    CAM_SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".mp3", ".wav", ".ogg", ".m4a"):
        p = CAM_SOUNDS_DIR / f"{cam_id}{ext}"
        if p.exists():
            return p
    return None
