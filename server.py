import json
import threading
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, Response, jsonify, send_from_directory, request, abort
from verifier import FaceVerifier
from face_db import load_all, save_face, delete_face, face_names, FACES_DIR
from camera_manager import (detect_cameras, load_config, save_config,
                             get_cam_sound_path, CAM_SOUNDS_DIR)

SETTINGS_FILE = Path("settings.json")


def load_settings():
    if not SETTINGS_FILE.exists():
        return {"general_sound_type": "notification"}
    with open(SETTINGS_FILE) as f:
        return json.load(f)


def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)


def get_general_sound_path():
    for ext in (".mp3", ".wav", ".ogg", ".m4a"):
        p = Path("static") / f"general_sound{ext}"
        if p.exists():
            return p
    return None

app = Flask(__name__, static_folder="static")

verifier    = FaceVerifier()
THRESHOLD   = 0.4
SKIP_FRAMES = 5
ENROLL_N    = 5

# ── shared face DB ────────────────────────────────────────────────────────────
faces      = load_all()
faces_lock = threading.Lock()

# ── per-camera state ──────────────────────────────────────────────────────────
# cam_states[id] = { detection: {...}, frame: ndarray|None, lock: Lock }
cam_states = {}
cam_config = load_config()

# ── enrollment (one at a time, any camera) ────────────────────────────────────
enroll = {
    "active": False, "name": "", "cam_id": None,
    "progress": 0, "samples": [], "best_frame": None,
    "sound_bytes": None, "sound_ext": None,
}
enroll_lock = threading.Lock()


# ── helpers ───────────────────────────────────────────────────────────────────
def blank_detection():
    return {"face_detected": False, "matched_name": None, "score": 0.0}


def camera_loop(cam_id):
    global faces
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    state = cam_states[cam_id]
    n = 0
    last_name, last_score, last_bbox = None, 0.0, None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        with enroll_lock:
            my_turn = enroll["active"] and enroll["cam_id"] == cam_id
            e_name  = enroll["name"]
            e_prog  = enroll["progress"]

        if my_turn:
            emb, bbox = verifier.get_embedding(frame)
            if emb is not None:
                with enroll_lock:
                    enroll["samples"].append(emb)
                    enroll["progress"] = len(enroll["samples"])
                    if enroll["best_frame"] is None and bbox is not None:
                        x1, y1, x2, y2 = bbox
                        enroll["best_frame"] = frame[max(0, y1):y2, max(0, x1):x2].copy()
                    if len(enroll["samples"]) >= ENROLL_N:
                        _finish_enrollment()

            cv2.putText(frame,
                        f"Enrolling {e_name}: {e_prog}/{ENROLL_N}  — keep still",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 220), 2)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)
        else:
            if n % SKIP_FRAMES == 0:
                with faces_lock:
                    fs = list(faces)
                emb, bbox = verifier.get_embedding(frame)
                last_name, last_score = verifier.verify_against_all(emb, fs, THRESHOLD)
                last_bbox = bbox
                with state["lock"]:
                    state["detection"]["face_detected"] = bbox is not None
                    state["detection"]["matched_name"]  = last_name
                    state["detection"]["score"]         = last_score

            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                color = (0, 255, 0) if last_name else (0, 0, 200)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame,
                            f"{last_name or 'Unknown'} ({last_score:.2f})",
                            (x1, max(y1 - 10, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        n += 1
        with state["lock"]:
            state["frame"] = frame.copy()

    cap.release()


def _finish_enrollment():
    global faces
    name       = enroll["name"]
    samples    = enroll["samples"]
    best_frame = enroll["best_frame"]
    s_bytes    = enroll["sound_bytes"]
    s_ext      = enroll["sound_ext"]

    emb = np.mean(samples, axis=0)
    emb /= np.linalg.norm(emb)
    photo = best_frame if best_frame is not None else np.zeros((100, 100, 3), np.uint8)
    save_face(name, emb, photo, s_bytes, s_ext)

    enroll.update({"active": False, "samples": [], "progress": 0,
                   "best_frame": None, "sound_bytes": None, "sound_ext": None})
    with faces_lock:
        faces = load_all()


def mjpeg(cam_id):
    state = cam_states[cam_id]
    while True:
        with state["lock"]:
            f = state["frame"]
        if f is None:
            continue
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


# ── startup: detect cameras, launch threads ───────────────────────────────────
available = detect_cameras()

for cid in available:
    cam_states[cid] = {"detection": blank_detection(), "frame": None, "lock": threading.Lock()}
    cam_config.setdefault(cid, {"label": f"Camera {cid}"})

save_config(cam_config)

for cid in available:
    threading.Thread(target=camera_loop, args=(cid,), daemon=True).start()


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/video_feed/<int:cam_id>")
def video_feed(cam_id):
    if cam_id not in cam_states:
        abort(404)
    return Response(mjpeg(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cameras")
def list_cameras():
    out = []
    for cid in available:
        cfg = cam_config.get(cid, {})
        out.append({
            "id": cid,
            "label": cfg.get("label", f"Camera {cid}"),
            "has_sound": get_cam_sound_path(cid) is not None,
        })
    return jsonify(out)


@app.route("/cameras/<int:cam_id>/label", methods=["POST"])
def set_label(cam_id):
    label = (request.json or {}).get("label", "").strip()
    if not label:
        return jsonify({"error": "label required"}), 400
    cam_config.setdefault(cam_id, {})["label"] = label
    save_config(cam_config)
    return jsonify({"ok": True})


@app.route("/cameras/<int:cam_id>/sound", methods=["POST"])
def upload_cam_sound(cam_id):
    if "sound" not in request.files:
        return jsonify({"error": "no file"}), 400
    f   = request.files["sound"]
    ext = Path(f.filename).suffix.lower() if f.filename else ".mp3"
    CAM_SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    for old in CAM_SOUNDS_DIR.glob(f"{cam_id}.*"):
        old.unlink()
    f.save(CAM_SOUNDS_DIR / f"{cam_id}{ext}")
    return jsonify({"ok": True})


@app.route("/cam_sound/<int:cam_id>")
def cam_sound(cam_id):
    p = get_cam_sound_path(cam_id)
    if not p:
        abort(404)
    return send_from_directory(p.parent, p.name)


@app.route("/status")
def status():
    with enroll_lock:
        e = {"active": enroll["active"], "name": enroll["name"],
             "cam_id": enroll["cam_id"], "progress": enroll["progress"]}

    cameras = []
    for cid in available:
        s = cam_states[cid]
        with s["lock"]:
            d = dict(s["detection"])
        cfg  = cam_config.get(cid, {})
        name = d.get("matched_name")

        # person's custom sound if set; otherwise JS uses TTS as general voice
        per_snd = None
        if name:
            for ext in (".mp3", ".wav", ".ogg", ".m4a"):
                if (FACES_DIR / name / f"sound{ext}").exists():
                    per_snd = f"/faces/{name}/sound"
                    break
        sound_url = per_snd

        cameras.append({
            "cam_id":       cid,
            "label":        cfg.get("label", f"Camera {cid}"),
            "face_detected":d["face_detected"],
            "matched_name": name,
            "score":        d["score"],
            "sound_url":    sound_url,
        })

    return jsonify({"cameras": cameras, "enroll": e})


@app.route("/faces")
def list_faces_route():
    return jsonify([{"name": n, "photo_url": f"/faces/{n}/photo"} for n in face_names()])


@app.route("/faces/<name>/photo")
def face_photo(name):
    p = FACES_DIR / name / "photo.jpg"
    if not p.exists():
        abort(404)
    return send_from_directory(p.parent, "photo.jpg")


@app.route("/faces/<name>/sound")
def face_sound(name):
    d = FACES_DIR / name
    for ext in (".mp3", ".wav", ".ogg", ".m4a"):
        p = d / f"sound{ext}"
        if p.exists():
            return send_from_directory(p.parent, p.name)
    abort(404)


@app.route("/faces/<name>", methods=["DELETE"])
def remove_face(name):
    global faces
    delete_face(name)
    with faces_lock:
        faces = load_all()
    return jsonify({"ok": True})


@app.route("/enroll", methods=["POST"])
def start_enroll():
    name   = request.form.get("name", "").strip()
    cam_id = int(request.form.get("cam_id", available[0]))
    if not name:
        return jsonify({"error": "name required"}), 400
    if cam_id not in cam_states:
        return jsonify({"error": "invalid camera"}), 400

    s_bytes, s_ext = None, None
    if "sound" in request.files:
        f      = request.files["sound"]
        s_bytes = f.read()
        s_ext   = Path(f.filename).suffix.lower() if f.filename else ".mp3"

    with enroll_lock:
        if enroll["active"]:
            return jsonify({"error": "enrollment already in progress"}), 409
        enroll.update({"active": True, "name": name, "cam_id": cam_id,
                       "progress": 0, "samples": [], "best_frame": None,
                       "sound_bytes": s_bytes, "sound_ext": s_ext})

    return jsonify({"ok": True, "name": name, "cam_id": cam_id})


@app.route("/enroll/image", methods=["POST"])
def enroll_from_image():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400

    img_bytes = request.files["image"].read()
    arr   = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid image file"}), 400

    emb, bbox = verifier.get_embedding(frame)
    if emb is None:
        return jsonify({"error": "no face detected in the image"}), 400

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        photo = frame[max(0, y1):y2, max(0, x1):x2]
    else:
        photo = frame

    s_bytes, s_ext = None, None
    if "sound" in request.files:
        f       = request.files["sound"]
        s_bytes = f.read()
        s_ext   = Path(f.filename).suffix.lower() if f.filename else ".mp3"

    save_face(name, emb, photo, s_bytes, s_ext)
    global faces
    with faces_lock:
        faces = load_all()

    return jsonify({"ok": True, "name": name})


@app.route("/settings")
def get_settings():
    s = load_settings()
    s["has_custom_sound"] = get_general_sound_path() is not None
    return jsonify(s)


@app.route("/settings", methods=["POST"])
def update_settings():
    s = load_settings()
    data = request.json or {}
    if "general_sound_type" in data:
        s["general_sound_type"] = data["general_sound_type"]
    save_settings(s)
    return jsonify({"ok": True})


@app.route("/settings/general-sound", methods=["POST"])
def upload_general_sound():
    if "sound" not in request.files:
        return jsonify({"error": "no file"}), 400
    f   = request.files["sound"]
    ext = Path(f.filename).suffix.lower() if f.filename else ".mp3"
    for old in Path("static").glob("general_sound.*"):
        old.unlink()
    f.save(Path("static") / f"general_sound{ext}")
    return jsonify({"ok": True})


@app.route("/general_sound")
def general_sound():
    p = get_general_sound_path()
    if not p:
        abort(404)
    return send_from_directory(p.parent, p.name)


if __name__ == "__main__":
    print(f"Cameras found: {available}")
    print("Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
