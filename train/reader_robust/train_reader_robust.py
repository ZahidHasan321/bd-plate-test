"""Make the plate reader robust to blur and compression (Kaggle GPU notebook).

Same data as train_kaggle.py (NightLPBD crops + LPDB-A crops, NightLPBD test vehicles excluded), plus one degraded
copy of every training crop: random motion blur, JPEG quality 30-80, and a shrink-then-enlarge step. Starts from the
night-v1 readers (release of this repo). Outputs in /kaggle/working/out: read_640.pt, read_320.pt, summary.json

Original description of the shared data preparation:
Fine-tune the avro-s pipeline (plate detector + character reader) on Bangladeshi night plates.

Runs as a Kaggle GPU script (see push.sh). Inputs, attached as Kaggle datasets:
  - NightLPBD (night images, plate + character boxes, CC-BY-4.0)
  - Bangla LPDB-A plate crops (102 character/district classes, same order as Avro's reader, CC-BY-4.0)
Starting weights are Avro's YOLOv8 models, downloaded from GitHub.

NightLPBD test-set vehicles are removed from training, so bench.py on the NightLPBD test split stays honest.
Outputs in /kaggle/working/out: detect.pt, read_640.pt, read_320.pt, summary.json
"""

import csv
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3.76"], check=True)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

WORK = Path("/kaggle/working")
DATA, OUT = WORK / "data", WORK / "out"
OUT.mkdir(parents=True, exist_ok=True)

# Avro reader classes (identical order to LPDB-A's 102 classes).
NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Metro', 'A', 'Bha', 'Cha', 'Chha', 'Da', 'DA', 'E',
         'Ga', 'Gha', 'Ha', 'Ja', 'Jha', 'Ka', 'Kha', 'La', 'Ma', 'Na', 'Pa', 'Sa', 'Sha', 'Ta', 'THA', 'Tha', 'U',
         'Bagerhat', 'Bagura', 'Bandarban', 'Barguna', 'Barisal', 'Bhola', 'Brahmanbaria', 'Chandpur',
         'Chapainawabganj', 'Chatto', 'Chattogram', 'Chuadanga', 'Coxs Bazar', 'Cumilla', 'Dhaka', 'Dinajpur',
         'Faridpur', 'Feni', 'Gaibandha', 'Gazipur', 'Gopalganj', 'Habiganj', 'Jamalpur', 'Jessore', 'Jhalokati',
         'Jhenaidah', 'Joypurhat', 'Khagrachari', 'Khulna', 'Kishoreganj', 'Kurigram', 'Kustia', 'Lakshmipur',
         'Lalmonirhat', 'Madaripur', 'Magura', 'Manikganj', 'Meherpur', 'Moulvibazar', 'Mymensingh', 'Naogaon',
         'Narail', 'Narayanganj', 'Narsingdi', 'Natore', 'Netrokona', 'Nilphamari', 'Noakhali', 'Pabna',
         'panchagarh', 'Patuakhali', 'Pirojpur', 'Raj', 'Rajbari', 'Rajshahi', 'Rangamati', 'Rangpur', 'Satkhira',
         'Shariatpur', 'Sherpur', 'Sirajganj', 'Sunamganj', 'Sylhet', 'Tangail', 'Thakurgaon', 'Dha', 'Ba']
IDX = {n: i for i, n in enumerate(NAMES)}

# NightLPBD label -> Avro class. Whole words and letters only.
NIGHT_MAP = {**{str(d): str(d) for d in range(10)},
             "Dhaka": "Dhaka", "meTrO": "Metro", "khulona": "Khulna", "zoshOro": "Jessore",
             "bo": "Ba", "co": "Cha", "go": "Ga", "ho": "Ha", "ka": "Ka", "lo": "La", "na": "Na", "To": "Ta"}
WORDS = {"Dhaka", "meTrO", "khulona", "zoshOro"}
PIECES = {"Dha", "khu", "me", "TrO", "zo", "shO", "ro"}  # syllables of the words above
AMBIGUOUS = {"ka", "na"}  # a letter on its own, a syllable when inside Dhaka / khulona

REL = "https://github.com/ZahidHasan321/bd-plate-test/releases/download/night-v1"
for name in ("read", "read_320"):
    urllib.request.urlretrieve(f"{REL}/night_{name}.pt", WORK / f"start_{name}.pt")


def find_one(pattern, check=lambda p: True):
    hits = [Path(p) for p in glob.glob(pattern, recursive=True) if check(Path(p))]
    if not hits:
        raise SystemExit(f"not found: {pattern}")
    return hits[0]


# ------------------------------------------------------------------ NightLPBD
night = find_one("/kaggle/input/**/data.yaml", lambda p: "zoshOro" in p.read_text()).parent
night_names = (night / "classes.txt").read_text().split()
print("NightLPBD at", night)


def plate_number(filename):
    m = re.match(r"^[A-Za-z_]+?_*(\d{2})_(\d{4})", filename)
    return m.group(1) + m.group(2) if m else None


test_plates = {plate_number(r["file_name"]) for r in csv.DictReader(open(night / "ids_test.csv"))}
stats = {"leaky_dropped": 0, "night_crops": {}, "night_skipped": 0}


def yolo_rows(path):
    return [(night_names[int(r[0])], *map(float, r[1:5])) for r in (l.split() for l in open(path)) if r]


def inside(c, box):
    _, x, y, *_ = c
    _, bx, by, bw, bh = box
    return abs(x - bx) <= bw / 2 and abs(y - by) <= bh / 2


for split, dst in (("train", "train"), ("valid", "val")):
    ids = list(csv.DictReader(open(night / f"ids_{split}.csv")))
    kept = [r for r in ids if plate_number(r["file_name"]) not in test_plates]
    stats["leaky_dropped"] += len(ids) - len(kept)
    for d in ("det", "read"):
        for sub in ("images", "labels"):
            (DATA / d / dst / sub).mkdir(parents=True, exist_ok=True)
    n_crops = 0
    for r in kept:
        img_path = night / split / "images" / f"{r['file_id']}.jpg"
        lab_path = night / split / "labels" / f"{r['file_id']}.txt"
        if not img_path.exists() or not lab_path.exists():
            continue
        rows = yolo_rows(lab_path)
        plates = [b for b in rows if b[0] == "plate"]
        # Detector: plate boxes on the full image (downscaled to keep the dataset small).
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        s = min(1.0, 1280 / max(h, w))
        cv2.imwrite(str(DATA / "det" / dst / "images" / img_path.name), cv2.resize(img, (round(w * s), round(h * s))))
        with open(DATA / "det" / dst / "labels" / lab_path.name, "w") as f:
            for _, x, y, bw, bh in plates:
                f.write(f"0 {x} {y} {bw} {bh}\n")
        # Reader: crop each plate, keep whole words / letters / digits, re-express boxes in crop coords.
        words = [b for b in rows if b[0] in WORDS]
        for k, (_, px, py, pw, ph) in enumerate(plates):
            x1, x2 = max(0, (px - pw * 0.6) * w), min(w, (px + pw * 0.6) * w)
            y1, y2 = max(0, (py - ph * 0.65) * h), min(h, (py + ph * 0.65) * h)
            cw, ch = x2 - x1, y2 - y1
            out, ok = [], True
            for c in rows:
                name, x, y, bw, bh = c
                if not (x1 <= x * w <= x2 and y1 <= y * h <= y2) or name == "plate":
                    continue
                if name in PIECES:
                    ok &= any(inside(c, wd) for wd in words)  # a syllable with no word box -> incomplete label
                    continue
                if name in AMBIGUOUS and any(inside(c, wd) for wd in words):
                    continue
                if name not in NIGHT_MAP:
                    ok = False
                    continue
                out.append(f"{IDX[NIGHT_MAP[name]]} {(x * w - x1) / cw:.6f} {(y * h - y1) / ch:.6f} "
                           f"{bw * w / cw:.6f} {bh * h / ch:.6f}\n")
            if not ok or not out:
                stats["night_skipped"] += 1
                continue
            stem = f"night_{r['file_id']}_{k}"
            cv2.imwrite(str(DATA / "read" / dst / "images" / f"{stem}.jpg"), img[int(y1):int(y2), int(x1):int(x2)])
            (DATA / "read" / dst / "labels" / f"{stem}.txt").write_text("".join(out))
            n_crops += 1
    stats["night_crops"][dst] = n_crops

# ------------------------------------------------------------------ LPDB-A crops (daytime, keeps old skills)
lpdb_yaml = find_one("/kaggle/input/**/data.yaml", lambda p: "Chapainawabganj" in p.read_text() and "nc: 102" in p.read_text())
stats["lpdb"] = {}
for src_split, dst in (("train", "train"), ("val", "val"), ("valid", "val")):
    img_dir = lpdb_yaml.parent / src_split / "images"
    if not img_dir.exists():
        continue
    n = 0
    for img in img_dir.iterdir():
        lab = lpdb_yaml.parent / src_split / "labels" / f"{img.stem}.txt"
        if lab.exists() and lab.stat().st_size:
            shutil.copy(img, DATA / "read" / dst / "images" / f"lpdb_{img.name}")
            shutil.copy(lab, DATA / "read" / dst / "labels" / f"lpdb_{img.stem}.txt")
            n += 1
    stats["lpdb"][src_split] = n
print(json.dumps(stats, indent=1))



def degrade(img, rng):
    """What phone/CCTV video does to a plate: motion blur, heavy JPEG, lost resolution."""
    h, w = img.shape[:2]
    k = int(rng.integers(3, 16))
    kernel = np.zeros((k, k), np.float32)
    angle = rng.uniform(0, 180)
    c = (k - 1) / 2
    for t in np.linspace(-c, c, 2 * k):
        x, y = int(round(c + t * np.cos(np.radians(angle)))), int(round(c + t * np.sin(np.radians(angle))))
        kernel[min(max(y, 0), k - 1), min(max(x, 0), k - 1)] = 1
    img = cv2.filter2D(img, -1, kernel / kernel.sum())
    s = rng.uniform(0.25, 0.8)
    img = cv2.resize(cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA),
                     (w, h), interpolation=cv2.INTER_LINEAR)
    q = int(rng.integers(30, 81))
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


rng = np.random.default_rng(0)
n_aug = 0
for img_path in sorted(p for p in (DATA / "read" / "train" / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
    img = cv2.imread(str(img_path))
    lab = DATA / "read" / "train" / "labels" / f"{img_path.stem}.txt"
    if img is None or not lab.exists():
        continue
    cv2.imwrite(str(img_path.with_name(f"{img_path.stem}_deg.jpg")), degrade(img, rng))
    shutil.copy(lab, lab.with_name(f"{img_path.stem}_deg.txt"))
    n_aug += 1
stats["degraded_copies"] = n_aug
print(json.dumps(stats, indent=1), flush=True)

(DATA / "read.yaml").write_text(f"path: {DATA / 'read'}\ntrain: train/images\nval: val/images\nnames: {json.dumps(NAMES)}\n")

# ------------------------------------------------------------------ training
COMMON = dict(epochs=60, patience=15, lr0=0.003, fliplr=0.0, degrees=5, hsv_v=0.5, workers=4,
              project=str(WORK / "runs"), plots=False)
results = {}
for name, weights, data, imgsz in (("read_640", "start_read.pt", "read.yaml", 640),
                                   ("read_320", "start_read_320.pt", "read.yaml", 320)):
    model = YOLO(str(WORK / weights))
    model.train(data=str(DATA / data), imgsz=imgsz, batch=32, name=name, **COMMON)
    best = WORK / "runs" / name / "weights" / "best.pt"
    shutil.copy(best, OUT / f"{name}.pt")
    m = YOLO(str(best)).val(data=str(DATA / data), imgsz=imgsz, plots=False)
    results[name] = {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4)}
    print(name, results[name], flush=True)

(OUT / "summary.json").write_text(json.dumps({"data": stats, "val": results}, indent=1))
print(json.dumps(results, indent=1))
