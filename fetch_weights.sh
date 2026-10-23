#!/usr/bin/env bash
# Download the third-party Bangla plate models and convert them to ONNX (weights/onnx/).
# The models are not stored in this repo: several have no licence. Internal testing only.
set -e
cd "$(dirname "$0")"

get() {  # repo, path-in-repo, local destination
  [ -f "$3" ] && return
  mkdir -p "$(dirname "$3")"
  echo "downloading $1/$2"
  curl -fsSL -o "$3" "https://raw.githubusercontent.com/$1/HEAD/$(printf '%s' "$2" | sed 's/ /%20/g')"
}

get AvroHabib/Bengali_License_Plate_Recognition model-n-detection/best.pt weights/raw/avro/detect_n.pt
get AvroHabib/Bengali_License_Plate_Recognition model-n-ocr/best.pt       weights/raw/avro/read_n.pt
get AvroHabib/Bengali_License_Plate_Recognition model-s-ocr/best.pt       weights/raw/avro/read_s.pt
get prosensarker4/BLPD LPD_Model.pt  weights/raw/blpd/detect.pt
get prosensarker4/BLPD LPNR_Model.pt weights/raw/blpd/read.pt
get MaidulHasan/Project_BLPR model_lp_bb_detection/lp_bb_trained_weights/train/weights/best.pt       weights/raw/maidul/detect.pt
get MaidulHasan/Project_BLPR model_blp_text_extraction/blp_text_trained_weights/train/weights/best.pt weights/raw/maidul/read.pt
get rashedulalbab253/bangla-number-plate-detection-yolov11 license_plate_detection/yolov11_training/weights/best.pt weights/raw/rashed/read.pt
get zarifbinhasnat/a-better-model-for-dhaka-city work/claude/models/plate_B_yolo26s_run1.pt weights/raw/dhaka/detect.pt
# Our fine-tuned avro-s (NightLPBD + LPDB-A), published as a release of this repo.
REL=https://github.com/ZahidHasan321/bd-plate-test/releases/download/night-v1
for f in detect read read_320; do
  [ -f weights/raw/night/$f.pt ] || { mkdir -p weights/raw/night; echo "downloading night/$f.pt";
    curl -fsSL -o weights/raw/night/$f.pt "$REL/night_$f.pt"; }
done
# Plate detector further trained on night toll-booth CCTV.
[ -f weights/raw/cctv/detect.pt ] || { mkdir -p weights/raw/cctv; echo "downloading cctv/detect.pt";
  curl -fsSL -o weights/raw/cctv/detect.pt https://github.com/ZahidHasan321/bd-plate-test/releases/download/cctv-v1/cctv_detect.pt; }
[ -f weights/raw/coco/vehicles.pt ] || { mkdir -p weights/raw/coco; curl -fsSL -o weights/raw/coco/vehicles.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt; }

# One-time conversion; only this step needs PyTorch (CPU build).
YOLO_OFFLINE=1 uv run --group export python export.py
echo "Done. Start the app with ./run.sh (same Wi-Fi) or ./run.sh tunnel (any network)."
