"""Plate-recognition test service: phone browser -> this PC (CPU) -> boxes + text.

Pipelines (detector + reader) are defined in pipelines.yaml and switchable from the page.
Frames sent with `source=phone|video` also go through the vehicle tracker (tracker.py): one row per vehicle
in events.db, shown in the page's Vehicles list (/vehicles) with confirm / correct (/vehicles/<id>).
"""

import base64
import csv
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from engine import load_pipelines
from tracker import PlateTracker, connect

ROOT = Path(__file__).parent
CAPTURES = ROOT / "captures"
CAPTURES.mkdir(exist_ok=True)
LOG = ROOT / "results.csv"

pipelines = load_pipelines()
templates = Jinja2Templates(directory=ROOT / "templates")
app = FastAPI()
(ROOT / "events").mkdir(exist_ok=True)
app.mount("/events", StaticFiles(directory=ROOT / "events"), name="events")
app.mount("/samples", StaticFiles(directory=ROOT / "samples"), name="samples")
_last_save: dict[str, float] = {}
db = connect()
db_lock = threading.Lock()
trackers: dict[str, PlateTracker] = {}


def decode(image: UploadFile) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    return img


def record(result: dict) -> dict:
    """Save crops + a CSV row per plate (at most once per second per pipeline), strip crops for JSON."""
    name = result["pipeline"]
    now = time.time()
    save = result["plates"] and now - _last_save.get(name, 0) >= 1.0
    if save:
        _last_save[name] = now
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        new = not LOG.exists()
        with LOG.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "pipeline", "text", "bangla", "valid", "det_conf", "read_conf",
                            "detect_ms", "read_ms", "crop"])
            for i, p in enumerate(result["plates"]):
                crop = CAPTURES / name / f"{stamp}_{i}.jpg"
                crop.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(crop), p["crop"])
                w.writerow([stamp, name, p["text"], p["bangla"], p["valid"], p["det_conf"], p["conf"],
                            result["ms"]["detect"], result["ms"]["read"], crop.relative_to(ROOT)])
    for p in result["plates"]:
        # Small JPEG of the plate for the page (shown next to its reading).
        p["crop_jpg"] = base64.b64encode(cv2.imencode(".jpg", p.pop("crop"), [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
    return result


GROUPS = [("recommended", "★ Recommended"), ("fast", "Faster"), ("alternative", "Other good options"),
          ("reference", "Reference (older / heavy)"), ("baseline", "Baseline (not for Bangla)")]


@app.get("/")
def index(request: Request):
    groups = [(label, [p for p in pipelines.values() if p.tag == tag]) for tag, label in GROUPS]
    samples = sorted(f.name for f in (ROOT / "samples" / "photos").iterdir() if not f.name.startswith("."))
    return templates.TemplateResponse(request, "index.html", {"groups": groups, "samples": samples})


@app.post("/predict")
def predict(image: UploadFile, pipeline: str, source: str | None = None):
    if pipeline not in pipelines:
        raise HTTPException(404, f"unknown pipeline {pipeline}")
    img = decode(image)
    h, w = img.shape[:2]
    tracker = None
    if source in ("phone", "video"):
        with db_lock:
            tracker = trackers.setdefault(source, PlateTracker(source, db=db, log=lambda line: None))
    result = pipelines[pipeline].run(img, should_read=tracker.wants_read if tracker else None)
    if tracker:
        with db_lock:
            tracker.update(time.time(), img.shape, result["plates"])
    return {"width": w, "height": h, **record(result)}


VEHICLE_FIELDS = ("id", "camera", "plate", "bangla", "status", "votes", "readings", "first_seen", "last_seen",
                  "crop", "reviewed_plate", "reviewed")


@app.get("/vehicles")
def vehicles(source: str | None = None, limit: int = 50):
    """One row per vehicle, newest first (includes vehicles still on screen once confidently read)."""
    with db_lock:
        rows = db.execute("SELECT * FROM events WHERE (? IS NULL OR camera=?) ORDER BY last_seen DESC LIMIT ?",
                          (source, source, limit)).fetchall()
    return [{k: r[k] for k in VEHICLE_FIELDS} for r in rows]


@app.post("/vehicles/{event_id}")
def review(event_id: str, plate: str = Form("")):
    """Confirm (empty plate) or correct a vehicle's plate."""
    with db_lock:
        row = db.execute("SELECT plate FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "unknown vehicle")
        db.execute("UPDATE events SET reviewed=1, reviewed_plate=?, version=version+1 WHERE id=?",
                   (plate.strip() or row["plate"], event_id))
        db.commit()
    return {"ok": True}


# ---------------------------------------------------------------- uploaded videos, analysed on this machine
UPLOADS = ROOT / "uploads"
jobs: dict[str, dict] = {}


def analyse(job: dict, path: Path, pipeline, fps: float = 10.0):
    """Read the uploaded video here (no per-frame upload): ~`fps` frames per second of video, plates found on
    every one, read selectively; results are kept per frame time so the page can draw them in sync."""
    cap = cv2.VideoCapture(str(path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    job["duration"] = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / video_fps
    step, n, start = max(1, round(video_fps / fps)), 0, time.time()
    with db_lock:
        tracker = trackers["video"] = PlateTracker("video", db=db, log=lambda line: None)
    try:
        while not job["cancel"]:
            ok = cap.grab()
            if not ok:
                break
            n += 1
            if n % step:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = n / video_fps
            res = pipeline.run(frame, should_read=tracker.wants_read)
            with db_lock:
                tracker.update(start + t, frame.shape, res["plates"])
                boxes = [{"box": p["box"], **tracker.label(p["box"])} for p in res["plates"]]
            job["frames"].append({"t": round(t, 3), "w": frame.shape[1], "h": frame.shape[0], "plates": boxes})
            job["processed"] = t
        with db_lock:
            tracker.close_all()
    except Exception as e:  # report instead of dying silently
        job["error"] = str(e)
    finally:
        cap.release()
        job["done"] = True
        path.unlink(missing_ok=True)


@app.post("/video")
def upload_video(file: UploadFile = File(...), pipeline: str = "avro-s-cctv"):
    if pipeline not in pipelines:
        raise HTTPException(404, f"unknown pipeline {pipeline}")
    for j in jobs.values():
        j["cancel"] = True  # one video at a time
    UPLOADS.mkdir(exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{job_id}{Path(file.filename or 'v.mp4').suffix or '.mp4'}"
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    if not cv2.VideoCapture(str(path)).isOpened():
        path.unlink(missing_ok=True)
        raise HTTPException(400, "could not open this video (try MP4/H.264)")
    job = jobs[job_id] = {"frames": [], "processed": 0.0, "duration": 0.0, "done": False, "cancel": False, "error": None}
    threading.Thread(target=analyse, args=(job, path, pipelines[pipeline]), daemon=True).start()
    return {"job": job_id}


@app.get("/video/{job_id}")
def video_status(job_id: str, since: int = 0):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {"frames": job["frames"][since:], "next": len(job["frames"]), "processed": job["processed"],
            "duration": job["duration"], "done": job["done"], "error": job["error"]}


@app.delete("/video/{job_id}")
def cancel_video(job_id: str):
    if job_id in jobs:
        jobs[job_id]["cancel"] = True
    return {"ok": True}


@app.post("/close")
def close(source: str):
    """Source stopped (video ended / camera off): finalise vehicles still on screen."""
    with db_lock:
        tracker = trackers.pop(source, None)
        if tracker:
            tracker.close_all()
    return {"ok": True}


@app.delete("/vehicles")
def clear(source: str):
    """Start a fresh demo list for one source (rows are removed; crops stay on disk)."""
    with db_lock:
        db.execute("DELETE FROM events WHERE camera=?", (source,))
        db.commit()
        trackers.pop(source, None)
    return {"ok": True}


@app.post("/compare")
def compare(image: UploadFile):
    img = decode(image)
    h, w = img.shape[:2]
    return {"width": w, "height": h, "results": [record(p.run(img)) for p in pipelines.values()]}
