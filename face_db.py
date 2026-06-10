import shutil
import numpy as np
import cv2
from pathlib import Path

FACES_DIR = Path("faces")
_SOUND_EXTS = (".mp3", ".wav", ".ogg", ".m4a")


def save_face(name, embedding, photo_bgr, sound_bytes=None, sound_ext=None):
    d = FACES_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "embedding.npy", embedding)
    cv2.imwrite(str(d / "photo.jpg"), photo_bgr)
    if sound_bytes and sound_ext:
        with open(d / f"sound{sound_ext}", "wb") as f:
            f.write(sound_bytes)


def save_face_sound(name, sound_bytes, sound_ext, cam_id=None):
    """Save default sound (cam_id=None) or a per-camera override."""
    d = FACES_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    prefix = f"sound_cam_{cam_id}" if cam_id is not None else "sound"
    for old in d.glob(f"{prefix}.*"):
        old.unlink()
    with open(d / f"{prefix}{sound_ext}", "wb") as f:
        f.write(sound_bytes)


def get_face_sound_path(name, cam_id=None):
    """Return cam-specific sound path if set, else default, else None."""
    d = FACES_DIR / name
    if cam_id is not None:
        for ext in _SOUND_EXTS:
            p = d / f"sound_cam_{cam_id}{ext}"
            if p.exists():
                return p
    for ext in _SOUND_EXTS:
        p = d / f"sound{ext}"
        if p.exists():
            return p
    return None


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
        faces.append({
            "name": d.name,
            "embedding": np.load(emb_path),
            "photo": d / "photo.jpg",
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
