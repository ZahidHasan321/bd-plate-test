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

# Extra lower-resolution exports of readers (plate crops are small; 320 is ~4x cheaper than 640).
EXTRA = {"avro/read_s": [320]}

for pt in sorted(RAW.glob("*/*.pt")):
    for size in [640, *EXTRA.get(f"{pt.parent.name}/{pt.stem}", [])]:
        suffix = "" if size == 640 else f"_{size}"
        target = OUT / f"{pt.parent.name}_{pt.stem}{suffix}.onnx"
        if target.exists():
            print(f"skip {target.name}")
            continue
        model = YOLO(str(pt))
        # Fixed input size, NMS done in engine.py so the graph stays NPU-friendly.
        exported = model.export(format="onnx", imgsz=size, simplify=True, dynamic=False)
        shutil.move(exported, target)
        print(f"{pt} -> {target.name}  ({len(model.names)} classes)")
