"""Plate-recognition test service: phone browser -> this PC (CPU) -> boxes + text.

Pipelines (detector + reader) are defined in pipelines.yaml and switchable from the page.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates

from engine import load_pipelines

ROOT = Path(__file__).parent
CAPTURES = ROOT / "captures"
CAPTURES.mkdir(exist_ok=True)
LOG = ROOT / "results.csv"

pipelines = load_pipelines()
templates = Jinja2Templates(directory=ROOT / "templates")
app = FastAPI()
_last_save: dict[str, float] = {}


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
        del p["crop"]
    return result


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"pipelines": pipelines.values()})


@app.post("/predict")
def predict(image: UploadFile, pipeline: str):
    if pipeline not in pipelines:
        raise HTTPException(404, f"unknown pipeline {pipeline}")
    img = decode(image)
    h, w = img.shape[:2]
    return {"width": w, "height": h, **record(pipelines[pipeline].run(img))}


@app.post("/compare")
def compare(image: UploadFile):
    img = decode(image)
    h, w = img.shape[:2]
    return {"width": w, "height": h, "results": [record(p.run(img)) for p in pipelines.values()]}
