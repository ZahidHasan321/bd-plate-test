"""Plate-recognition test service: phone browser -> this PC (CPU) -> boxes + text.

Pipelines (detector + reader) are defined in pipelines.yaml and switchable from the page.
Frames sent with `source=phone|video` also go through the vehicle tracker (tracker.py): one row per vehicle
in events.db, shown in the page's Vehicles list (/vehicles) with confirm / correct (/vehicles/<id>).
"""

import base64
import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
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
    samples = {kind: sorted(f.name for f in (ROOT / "samples" / kind).iterdir() if not f.name.startswith("."))
               for kind in ("photos", "videos")}
    return templates.TemplateResponse(request, "index.html", {"groups": groups, "samples": samples})


@app.post("/predict")
def predict(image: UploadFile, pipeline: str, source: str | None = None):
    if pipeline not in pipelines:
        raise HTTPException(404, f"unknown pipeline {pipeline}")
    img = decode(image)
    h, w = img.shape[:2]
    result = pipelines[pipeline].run(img)
    if source in ("phone", "video"):
        with db_lock:
            tracker = trackers.setdefault(source, PlateTracker(source, db=db, log=lambda line: None))
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
