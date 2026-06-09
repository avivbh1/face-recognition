import shutil
import numpy as np
import cv2
from pathlib import Path

FACES_DIR = Path("faces")


def save_face(name, embedding, photo_bgr, sound_bytes=None, sound_ext=None):
    d = FACES_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "embedding.npy", embedding)
    cv2.imwrite(str(d / "photo.jpg"), photo_bgr)
    if sound_bytes and sound_ext:
        with open(d / f"sound{sound_ext}", "wb") as f:
            f.write(sound_bytes)


def load_all():
    faces = []
    if not FACES_DIR.exists():
        return faces
    for d in sorted(FACES_DIR.iterdir()):
        if not d.is_dir():
            continue
        emb_path = d / "embedding.npy"
        if not emb_path.exists():
            continue
        sound_path = None
        for ext in (".mp3", ".wav", ".ogg", ".m4a"):
            p = d / f"sound{ext}"
            if p.exists():
                sound_path = p
                break
        faces.append({
            "name": d.name,
            "embedding": np.load(emb_path),
            "photo": d / "photo.jpg",
            "sound": sound_path,
        })
    return faces


def delete_face(name):
    d = FACES_DIR / name
    if d.exists():
        shutil.rmtree(d)


def face_names():
    if not FACES_DIR.exists():
        return []
    return [d.name for d in sorted(FACES_DIR.iterdir())
            if d.is_dir() and (d / "embedding.npy").exists()]
