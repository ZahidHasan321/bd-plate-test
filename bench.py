"""Run pipelines over a folder of images; print readings, timings and (optionally) accuracy.

Usage:
  uv run python bench.py <image-folder> [-p pipeline ...] [--truth gt.csv] [--csv out.csv]

--truth: CSV with columns image, area, letter, digits (e.g. image=car1.jpg, area="DHAKA METRO",
         letter=GA, digits=123456). Each image is scored on its most confident plate.
Runs unchanged on an edge board with onnxruntime installed.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

from engine import load_pipelines

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument("-p", "--pipelines", nargs="*")
ap.add_argument("--truth", type=Path)
ap.add_argument("--csv", type=Path)
args = ap.parse_args()

pipelines = load_pipelines()
names = args.pipelines or list(pipelines)
images = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
truth = {r["image"]: r for r in csv.DictReader(args.truth.open())} if args.truth else {}
if truth:
    images = [p for p in images if p.name in truth]

rows, summary = [], []
for name in names:
    pipe = pipelines[name]
    pipe.run(cv2.imread(str(images[0])))  # warm-up: model load + first-run allocation
    stats = Counter()
    print(f"\n== {name}: {pipe.note}", flush=True)
    for path in images:
        res = pipe.run(cv2.imread(str(path)))
        stats["detect_ms"] += res["ms"]["detect"]
        stats["read_ms"] += res["ms"]["read"]
        best = max(res["plates"], key=lambda p: p["det_conf"], default=None)
        stats["found"] += best is not None
        stats["valid"] += bool(best and best["valid"])
        row = {"pipeline": name, "image": path.name, "text": best["text"] if best else "",
               "valid": bool(best and best["valid"]), "detect_ms": res["ms"]["detect"],
               "read_ms": res["ms"]["read"]}
        mark = ""
        if path.name in truth:
            gt = truth[path.name]
            digits_ok = bool(best) and best.get("digits") == gt["digits"]
            full_ok = digits_ok and best.get("area") == gt["area"] and best.get("letter") == gt["letter"]
            stats["digits_ok"] += digits_ok
            stats["full_ok"] += full_ok
            row.update(truth=f"{gt['area']}-{gt['letter']} {gt['digits']}", digits_ok=digits_ok, full_ok=full_ok)
            mark = "  ✓" if full_ok else "  ~digits" if digits_ok else f"  ✗ (truth {row['truth']})"
        rows.append(row)
        print(f"  {path.name[:28]:28}  {row['text'] or '-'}{mark}", flush=True)
    summary.append((name, stats, len(images)))

if args.csv:
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)

n = len(images)
head = f"\n{'pipeline':13} {'detect ms':>9} {'read ms':>8} {'found':>6} {'valid fmt':>9}"
print(head + (f" {'digits ok':>9} {'full ok':>8}" if truth else ""))
for name, s, n in summary:
    line = (f"{name:13} {s['detect_ms'] / n:9.0f} {s['read_ms'] / n:8.0f} {s['found'] / n:6.0%} "
            f"{s['valid'] / n:9.0%}")
    print(line + (f" {s['digits_ok'] / n:9.0%} {s['full_ok'] / n:8.0%}" if truth else ""))
print(f"({n} images)")
