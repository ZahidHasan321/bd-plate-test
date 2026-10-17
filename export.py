"""One-time conversion of every weights/raw/<repo>/<name>.pt to weights/onnx/<repo>_<name>.onnx.

Run with:  uv run --group export python export.py
Only this script needs torch/ultralytics; the app itself runs on onnxruntime alone.
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
RAW = ROOT / "weights" / "raw"
OUT = ROOT / "weights" / "onnx"
OUT.mkdir(parents=True, exist_ok=True)

for pt in sorted(RAW.glob("*/*.pt")):
    target = OUT / f"{pt.parent.name}_{pt.stem}.onnx"
    if target.exists():
        print(f"skip {target.name}")
        continue
    model = YOLO(str(pt))
    # Fixed 640 input, NMS done in engine.py so the graph stays NPU-friendly.
    exported = model.export(format="onnx", imgsz=640, simplify=True, dynamic=False)
    shutil.move(exported, target)
    print(f"{pt} -> {target.name}  ({len(model.names)} classes)")
