import json
import time
import uuid
import threading
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, Response, jsonify, send_from_directory, request, abort
from verifier import FaceVerifier
from face_db import (load_all, save_face, delete_face, face_names, FACES_DIR,
                     get_face_sound_path, save_face_sound)
from camera_manager import (detect_cameras, load_config, save_config,
                             get_cam_sound_path, CAM_SOUNDS_DIR)

SETTINGS_FILE   = Path("settings.json")
QUEUE_DIR       = Path("static") / "queue"
QUEUE_MAX       = 20
QUEUE_COOLDOWN  = 10.0   # seconds before same entity can appear again in queue


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
cam_states = {}
cam_config = load_config()

# ── enrollment ────────────────────────────────────────────────────────────────
enroll = {
    "active": False, "name": "", "cam_id": None,
    "progress": 0, "samples": [], "best_frame": None,
    "sound_bytes": None, "sound_ext": None,
}
enroll_lock = threading.Lock()

# ── event queue ───────────────────────────────────────────────────────────────
event_queue      = []          # newest first, max QUEUE_MAX items
queue_lock       = threading.Lock()
queue_last_seen  = {}          # (cam_id, entity_key) -> timestamp


def blank_detection():
    return {"face_detected": False, "matched_name": None, "score": 0.0}


# ── queue helper ──────────────────────────────────────────────────────────────
def push_queue_event(cam_id, cam_label, emb, matched_name, score, bbox, frame):
    entity = matched_name or "__unknown__"
    key    = (cam_id, entity)
    now    = time.time()

    with queue_lock:
        if now - queue_last_seen.get(key, 0) < QUEUE_COOLDOWN:
            return
        queue_last_seen[key] = now

    if bbox is None:
        return

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    pad = 20
    crop = frame[max(0, y1-pad):min(h, y2+pad),
                 max(0, x1-pad):min(w, x2+pad)].copy()

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    eid        = uuid.uuid4().hex[:12]
    crop_path  = QUEUE_DIR / f"{eid}.jpg"
    cv2.imwrite(str(crop_path), crop)

    event = {
        "id":           eid,
        "type":         "recognition" if matched_name else "detection",
        "timestamp":    now,
        "cam_id":       cam_id,
        "cam_label":    cam_label,
        "matched_name": matched_name,
        "score":        round(score, 3),
        "crop_url":     f"/queue/{eid}.jpg",
        "embedding":    emb.tolist() if emb is not None else None,
    }

    with queue_lock:
        event_queue.insert(0, event)
        while len(event_queue) > QUEUE_MAX:
            old = event_queue.pop()
            (QUEUE_DIR / f"{old['id']}.jpg").unlink(missing_ok=True)


# ── camera loop ───────────────────────────────────────────────────────────────
def camera_loop(cam_id):
    global faces
    cap   = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    state = cam_states[cam_id]
    n     = 0
    last_name, last_score, last_bbox, last_emb = None, 0.0, None, None

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

            cv2.putText(frame, f"Enrolling {e_name}: {e_prog}/{ENROLL_N}  — keep still",
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
                last_emb  = emb
                with state["lock"]:
                    state["detection"]["face_detected"] = bbox is not None
                    state["detection"]["matched_name"]  = last_name
                    state["detection"]["score"]         = last_score

                if bbox is not None:
                    cfg = cam_config.get(cam_id, {})
                    push_queue_event(cam_id, cfg.get("label", f"Camera {cam_id}"),
                                     last_emb, last_name, last_score, bbox, frame)

            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                color = (0, 255, 0) if last_name else (0, 0, 200)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{last_name or 'Unknown'} ({last_score:.2f})",
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


# ── startup ───────────────────────────────────────────────────────────────────
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
    return jsonify([{
        "id":        cid,
        "label":     cam_config.get(cid, {}).get("label", f"Camera {cid}"),
        "has_sound": get_cam_sound_path(cid) is not None,
    } for cid in available])


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

        sound_url = None
        if name:
            p = get_face_sound_path(name, cid)
            if p:
                if f"sound_cam_{cid}" in p.name:
                    sound_url = f"/faces/{name}/sound/{cid}"
                else:
                    sound_url = f"/faces/{name}/sound"

        cameras.append({
            "cam_id":        cid,
            "label":         cfg.get("label", f"Camera {cid}"),
            "face_detected": d["face_detected"],
            "matched_name":  name,
            "score":         d["score"],
            "sound_url":     sound_url,
        })

    return jsonify({"cameras": cameras, "enroll": e})


# ── faces ─────────────────────────────────────────────────────────────────────
@app.route("/faces")
def list_faces_route():
    result = []
    for n in face_names():
        has_default = get_face_sound_path(n) is not None
        cam_sounds  = {str(cid): get_face_sound_path(n, cid) is not None
                       for cid in available}
        result.append({
            "name":              n,
            "photo_url":         f"/faces/{n}/photo",
            "has_default_sound": has_default,
            "cam_sounds":        cam_sounds,
        })
    return jsonify(result)


@app.route("/faces/<name>/photo")
def face_photo(name):
    p = FACES_DIR / name / "photo.jpg"
    if not p.exists():
        abort(404)
    return send_from_directory(p.parent, "photo.jpg")


@app.route("/faces/<name>/sound", methods=["GET", "POST"])
def face_default_sound(name):
    if request.method == "POST":
        if "sound" not in request.files:
            return jsonify({"error": "no file"}), 400
        f   = request.files["sound"]
        ext = Path(f.filename).suffix.lower() if f.filename else ".mp3"
        save_face_sound(name, f.read(), ext, cam_id=None)
        return jsonify({"ok": True})
    # GET
    d = FACES_DIR / name
    for ext in (".mp3", ".wav", ".ogg", ".m4a"):
        p = d / f"sound{ext}"
        if p.exists():
            return send_from_directory(p.parent, p.name)
    abort(404)


@app.route("/faces/<name>/sound/<int:cam_id>", methods=["GET", "POST"])
def face_cam_sound(name, cam_id):
    if request.method == "POST":
        if "sound" not in request.files:
            return jsonify({"error": "no file"}), 400
        f   = request.files["sound"]
        ext = Path(f.filename).suffix.lower() if f.filename else ".mp3"
        save_face_sound(name, f.read(), ext, cam_id=cam_id)
        return jsonify({"ok": True})
    # GET
    d = FACES_DIR / name
    for ext in (".mp3", ".wav", ".ogg", ".m4a"):
        p = d / f"sound_cam_{cam_id}{ext}"
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


# ── queue ─────────────────────────────────────────────────────────────────────
@app.route("/queue")
def get_queue():
    with queue_lock:
        safe = [{k: v for k, v in e.items() if k != "embedding"}
                for e in event_queue]
    return jsonify(safe)


@app.route("/queue/<filename>")
def serve_queue_image(filename):
    return send_from_directory(QUEUE_DIR, filename)


@app.route("/enroll/queue/<event_id>", methods=["POST"])
def enroll_from_queue(event_id):
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400

    with queue_lock:
        event = next((e for e in event_queue if e["id"] == event_id), None)

    if not event:
        return jsonify({"error": "event not found"}), 404
    if not event.get("embedding"):
        return jsonify({"error": "no face data for this event"}), 400

    emb   = np.array(event["embedding"])
    crop  = cv2.imread(str(QUEUE_DIR / f"{event_id}.jpg"))
    photo = crop if crop is not None else np.zeros((100, 100, 3), np.uint8)

    save_face(name, emb, photo)
    global faces
    with faces_lock:
        faces = load_all()

    return jsonify({"ok": True, "name": name})


# ── enroll routes ─────────────────────────────────────────────────────────────
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
        f       = request.files["sound"]
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

    arr   = np.frombuffer(request.files["image"].read(), np.uint8)
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


# ── settings ──────────────────────────────────────────────────────────────────
@app.route("/settings")
def get_settings():
    s = load_settings()
    s["has_custom_sound"] = get_general_sound_path() is not None
    return jsonify(s)


@app.route("/settings", methods=["POST"])
def update_settings():
    s    = load_settings()
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
