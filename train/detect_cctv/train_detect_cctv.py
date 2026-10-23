"""Fine-tune the plate detector further on night toll-booth CCTV frames (Kaggle GPU notebook).

Starts from the night-v1 detector (release of this repo) and trains on NightLPBD plate boxes (test-set vehicles
excluded) + the private toll-booth CCTV train split (frames 1-194; 195-277 are held out locally for evaluation).
Output: /kaggle/working/out/detect_cctv.pt
"""

import csv
import glob
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3.76"], check=True)
import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

WORK = Path("/kaggle/working")
DATA, OUT = WORK / "data", WORK / "out"
OUT.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(
    "https://github.com/ZahidHasan321/bd-plate-test/releases/download/night-v1/night_detect.pt", WORK / "start.pt")

for split in ("train", "val"):
    for sub in ("images", "labels"):
        (DATA / split / sub).mkdir(parents=True, exist_ok=True)


def add(img_path: Path, boxes: list[str], split: str, prefix: str, copies: int = 1):
    """Downscale to <=1280 px and write image + plate-only labels."""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    h, w = img.shape[:2]
    s = min(1.0, 1280 / max(h, w))
    img = cv2.resize(img, (round(w * s), round(h * s)))
    for c in range(copies):
        stem = f"{prefix}_{img_path.stem}_{c}"
        cv2.imwrite(str(DATA / split / "images" / f"{stem}.jpg"), img)
        (DATA / split / "labels" / f"{stem}.txt").write_text("".join(boxes))
    return copies


# NightLPBD: plate class only, test-set vehicles excluded (same rule as the night-v1 job).
night = next(Path(p).parent for p in glob.glob("/kaggle/input/**/data.yaml", recursive=True)
             if "zoshOro" in Path(p).read_text())
names = (night / "classes.txt").read_text().split()
plate_no = lambda fn: (lambda m: m.group(1) + m.group(2) if m else None)(re.match(r"^[A-Za-z_]+?_*(\d{2})_(\d{4})", fn))
test_plates = {plate_no(r["file_name"]) for r in csv.DictReader(open(night / "ids_test.csv"))}
counts = {"night_train": 0, "night_val": 0, "cctv_train": 0}
for src, dst in (("train", "train"), ("valid", "val")):
    for r in csv.DictReader(open(night / f"ids_{src}.csv")):
        if plate_no(r["file_name"]) in test_plates:
            continue
        lab = night / src / "labels" / f"{r['file_id']}.txt"
        if not lab.exists():
            continue
        boxes = [f"0 {' '.join(l.split()[1:5])}\n" for l in open(lab) if l.split() and names[int(l.split()[0])] == "plate"]
        counts[f"night_{dst}"] += add(night / src / "images" / f"{r['file_id']}.jpg", boxes, dst, "night")

# Toll-booth CCTV train split (private dataset). Walk /kaggle/input following links; unzip any archive.
import os
cctv_root = None
for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
    if "tollbooth" not in root:
        continue
    print("cctv dataset:", root, dirs[:5], files[:5], flush=True)
    for f in files:
        if f.endswith(".zip"):
            zipfile.ZipFile(os.path.join(root, f)).extractall(WORK / "cctv")
            cctv_root = WORK / "cctv"
    if cctv_root is None and "labels" in dirs and "images" in dirs:
        cctv_root = Path(root)
if cctv_root is None:
    print(sorted(os.listdir("/kaggle/input")), flush=True)
for root, dirs, files in os.walk(cctv_root or "/nonexistent", followlinks=True):
    if Path(root).name != "labels":
        continue
    for f in files:
        lab = Path(root) / f
        img = next(iter((lab.parent.parent / "images").glob(f"{lab.stem}.*")), None)
        if img:
            # 3 copies: 193 CCTV frames vs ~720 NightLPBD images; augmentation makes the copies differ.
            counts["cctv_train"] += add(img, open(lab).readlines(), "train", "cctv", copies=3)
print(counts, flush=True)
if not counts["cctv_train"]:
    raise SystemExit("CCTV train split not found under /kaggle/input")

(DATA / "det.yaml").write_text(f"path: {DATA}\ntrain: train/images\nval: val/images\nnames: ['license_plate']\n")
model = YOLO(str(WORK / "start.pt"))
model.train(data=str(DATA / "det.yaml"), imgsz=640, batch=32, epochs=60, patience=15, lr0=0.003,
            fliplr=0.0, degrees=5, hsv_v=0.5, workers=4, project=str(WORK / "runs"), name="detect_cctv", plots=False)
shutil.copy(WORK / "runs/detect_cctv/weights/best.pt", OUT / "detect_cctv.pt")
print("done", counts)
