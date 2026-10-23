"""Plate pipelines on onnxruntime: plate detector -> plate crop -> reader -> plate text.

Bangla readers are YOLO character detectors: each digit / letter / district is a box.
We split the boxes into the plate's two lines, sort left to right and join the tokens.
"""

import ast
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

ROOT = Path(__file__).parent

# ---------------------------------------------------------------- vocabulary
# Model class names differ slightly between repos; map them to one spelling.
ALIASES = {
    "Cox's Bazar": "Coxs Bazar", "Cox-s Bazar": "Coxs Bazar", "Cox-s-Bazar": "Coxs Bazar",
    "Bazar": "Coxs Bazar", "Nawabganj": "Chapainawabganj", "Nowabganj": "Chapainawabganj",
    "panchagarh": "Panchagarh",
}

# Vehicle class letters. Ta = ট (LPDB-A "T"); Da/DA/THA are best guesses at the dataset transliteration.
LETTERS = {
    "Ka": "ক", "Kha": "খ", "Ga": "গ", "Gha": "ঘ", "Cha": "চ", "Chha": "ছ", "Ja": "জ", "Jha": "ঝ",
    "Ta": "ট", "Tha": "থ", "THA": "ঠ", "Dha": "ঢ", "Da": "দ", "DA": "ড", "Na": "ন", "Pa": "প",
    "Ba": "ব", "Bha": "ভ", "Ma": "ম", "La": "ল", "Sha": "শ", "Sa": "স", "Ha": "হ",
    "A": "অ", "E": "ই", "U": "উ",
}

AREAS = {
    "Metro": "মেট্রো", "Chatto": "চট্ট", "Raj": "রাজ",
    "Bagerhat": "বাগেরহাট", "Bagura": "বগুড়া", "Bandarban": "বান্দরবান", "Barguna": "বরগুনা",
    "Barisal": "বরিশাল", "Bhola": "ভোলা", "Brahmanbaria": "ব্রাহ্মণবাড়িয়া", "Chandpur": "চাঁদপুর",
    "Chapainawabganj": "চাঁপাইনবাবগঞ্জ", "Chattogram": "চট্টগ্রাম", "Chuadanga": "চুয়াডাঙ্গা",
    "Coxs Bazar": "কক্সবাজার", "Cumilla": "কুমিল্লা", "Dhaka": "ঢাকা", "Dinajpur": "দিনাজপুর",
    "Faridpur": "ফরিদপুর", "Feni": "ফেনী", "Gaibandha": "গাইবান্ধা", "Gazipur": "গাজীপুর",
    "Gopalganj": "গোপালগঞ্জ", "Habiganj": "হবিগঞ্জ", "Jamalpur": "জামালপুর", "Jessore": "যশোর",
    "Jhalokati": "ঝালকাঠি", "Jhenaidah": "ঝিনাইদহ", "Joypurhat": "জয়পুরহাট",
    "Khagrachari": "খাগড়াছড়ি", "Khulna": "খুলনা", "Kishoreganj": "কিশোরগঞ্জ",
    "Kurigram": "কুড়িগ্রাম", "Kustia": "কুষ্টিয়া", "Lakshmipur": "লক্ষ্মীপুর",
    "Lalmonirhat": "লালমনিরহাট", "Madaripur": "মাদারীপুর", "Magura": "মাগুরা",
    "Manikganj": "মানিকগঞ্জ", "Meherpur": "মেহেরপুর", "Moulvibazar": "মৌলভীবাজার",
    "Munshiganj": "মুন্সিগঞ্জ", "Mymensingh": "ময়মনসিংহ", "Naogaon": "নওগাঁ", "Narail": "নড়াইল",
    "Narayanganj": "নারায়ণগঞ্জ", "Narsingdi": "নরসিংদী", "Natore": "নাটোর",
    "Netrokona": "নেত্রকোনা", "Nilphamari": "নীলফামারী", "Noakhali": "নোয়াখালী", "Pabna": "পাবনা",
    "Panchagarh": "পঞ্চগড়", "Patuakhali": "পটুয়াখালী", "Pirojpur": "পিরোজপুর",
    "Rajbari": "রাজবাড়ী", "Rajshahi": "রাজশাহী", "Rangamati": "রাঙ্গামাটি", "Rangpur": "রংপুর",
    "Satkhira": "সাতক্ষীরা", "Shariatpur": "শরীয়তপুর", "Sherpur": "শেরপুর",
    "Sirajganj": "সিরাজগঞ্জ", "Sunamganj": "সুনামগঞ্জ", "Sylhet": "সিলেট", "Tangail": "টাঙ্গাইল",
    "Thakurgaon": "ঠাকুরগাঁও",
}

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def assemble(boxes: np.ndarray, tokens: list[str], scores: np.ndarray) -> dict:
    """Character boxes (x1,y1,x2,y2) + tokens -> plate text in Latin and Bangla, plus format check.

    Applies the BD plate grammar: one district (+ optional Metro), one class letter, six digits.
    When the reader returns extras, the most confident ones are kept.
    """
    if not tokens:
        return {"text": "", "bangla": "", "valid": False, "area": "", "letter": "", "digits": ""}
    tokens = [ALIASES.get(t, t) for t in tokens]
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    order = np.argsort(cy)
    gaps = np.diff(cy[order])
    height = np.median(boxes[:, 3] - boxes[:, 1])
    lines = [order]
    if len(gaps) and gaps.max() > 0.5 * height:
        cut = gaps.argmax() + 1
        lines = [order[:cut], order[cut:]]
    reading = [i for line in lines for i in sorted(line, key=lambda i: boxes[i, 0])]

    def best(candidates, k):
        """Top-k most confident candidates, kept in reading order."""
        keep = set(sorted(candidates, key=lambda i: -scores[i])[:k])
        return [i for i in candidates if i in keep]

    digits = "".join(tokens[i] for i in best([i for i in reading if tokens[i].isdigit()], 6))
    district = [tokens[i] for i in best([i for i in reading if tokens[i] in AREAS and tokens[i] != "Metro"], 1)]
    metro = ["Metro"] if "Metro" in tokens else []
    area = district + metro
    letter = [tokens[i] for i in best([i for i in reading if tokens[i] in LETTERS], 1)]

    num = f"{digits[:2]}-{digits[2:]}" if len(digits) == 6 else digits
    text = f"{' '.join(area).upper()}-{''.join(letter).upper()} {num}".strip(" -")
    bangla = f"{' '.join(AREAS[a] for a in area)}-{''.join(LETTERS[l] for l in letter)} {num.translate(BN_DIGITS)}"
    valid = bool(district and letter and len(digits) == 6)
    return {"text": text, "bangla": bangla.strip(" -"), "valid": valid,
            "area": " ".join(area).upper(), "letter": "".join(letter).upper(), "digits": digits}


# ---------------------------------------------------------------- YOLO on ONNX
@lru_cache(maxsize=10)
def session(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(str(ROOT / path), providers=["CPUExecutionProvider"])


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, left, top


def yolo(path: str, img: np.ndarray, conf: float, iou: float = 0.5, classes: list[int] | None = None):
    """Run a YOLOv8/11/12/26 ONNX detector (optionally only some classes). Returns boxes (N,4) in image pixels, scores, class ids, names."""
    sess = session(path)
    size = sess.get_inputs()[0].shape[2]
    names = ast.literal_eval(sess.get_modelmeta().custom_metadata_map["names"])
    canvas, r, left, top = letterbox(img, size)
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0].T  # (anchors, 4 + classes)
    scores = out[:, 4:].max(1)
    cls = out[:, 4:].argmax(1)
    keep = scores >= conf
    if classes is not None:
        keep &= np.isin(cls, classes)
    out, scores, cls = out[keep], scores[keep], cls[keep]
    cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    xywh = np.stack([cx - w / 2, cy - h / 2, w, h], 1)
    # Class-agnostic NMS: characters and plates should not overlap each other.
    idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou), dtype=int).reshape(-1)
    xyxy = np.stack([xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], 1)[idx]
    xyxy = (xyxy - [left, top, left, top]) / r
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, img.shape[1])
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, img.shape[0])
    return xyxy.astype(int), scores[idx], cls[idx], names


# ---------------------------------------------------------------- hub models (fast-alpr)
@lru_cache(maxsize=2)
def oim_detector(model: str):
    from fast_alpr.default_detector import DefaultDetector
    return DefaultDetector(model_name=model, conf_thresh=0.25)


@lru_cache(maxsize=2)
def fpo_reader(model: str):
    from fast_alpr.default_ocr import DefaultOCR
    return DefaultOCR(hub_ocr_model=model)


# ---------------------------------------------------------------- pipelines
@dataclass
class Pipeline:
    name: str
    note: str
    detector: dict
    reader: dict
    vehicles: dict | None = None
    tag: str = "reference"

    def detect(self, img):
        v = self.vehicles
        if not v:
            return self.detect_plates(img)
        # Cascade: find vehicles, look for plates inside each vehicle crop, map back to the frame.
        vboxes, _, _, _ = yolo(v["path"], img, v.get("conf", 0.2), classes=v.get("classes"))
        found = []
        for x1, y1, x2, y2 in vboxes:
            for (a, b, c, d), s in self.detect_plates(img[y1:y2, x1:x2]):
                found.append(((a + x1, b + y1, c + x1, d + y1), s))
        if not found:
            return []
        # Overlapping vehicle boxes can yield the same plate twice.
        xywh = [[a, b, c - a, d - b] for (a, b, c, d), _ in found]
        idx = np.array(cv2.dnn.NMSBoxes(xywh, [s for _, s in found], 0.0, 0.5), dtype=int).reshape(-1)
        return [found[i] for i in idx]

    def detect_plates(self, img):
        d = self.detector
        if d["type"] == "oim":
            dets = oim_detector(d["model"]).predict(img)
            return [((b.bounding_box.x1, b.bounding_box.y1, b.bounding_box.x2, b.bounding_box.y2),
                     float(b.confidence)) for b in dets]
        boxes, scores, _, _ = yolo(d["path"], img, d.get("conf", 0.25))
        return [(tuple(b), float(s)) for b, s in zip(boxes, scores)]

    def read(self, crop):
        rd = self.reader
        if rd["type"] == "fpo":
            res = fpo_reader(rd["model"]).predict(crop)
            if res is None:
                return {"text": "", "bangla": "", "valid": False, "conf": None}
            c = res.confidence
            c = sum(c) / len(c) if isinstance(c, list) and c else c
            return {"text": res.text, "bangla": "", "valid": False, "conf": round(float(c), 3)}
        boxes, scores, cls, names = yolo(rd["path"], crop, rd.get("conf", 0.25))
        out = assemble(boxes, [names[int(c)] for c in cls], scores)
        out["conf"] = round(float(scores.min()), 3) if len(scores) else None  # weakest character
        out["chars"] = len(scores)
        return out

    def run(self, img) -> dict:
        t0 = time.perf_counter()
        dets = self.detect(img)
        t1 = time.perf_counter()
        h, w = img.shape[:2]
        plates = []
        for (x1, y1, x2, y2), det_conf in dets:
            x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), w), min(int(y2), h)
            crop = img[y1:y2, x1:x2]
            if not crop.size:
                continue
            plates.append({"box": [x1, y1, x2, y2], "det_conf": round(det_conf, 3),
                           **self.read(crop), "crop": crop})
        t2 = time.perf_counter()
        return {"pipeline": self.name, "plates": plates,
                "ms": {"detect": round((t1 - t0) * 1000, 1), "read": round((t2 - t1) * 1000, 1)}}


def load_pipelines() -> dict[str, Pipeline]:
    cfg = yaml.safe_load((ROOT / "pipelines.yaml").read_text())["pipelines"]
    return {name: Pipeline(name=name, note=p.get("note", ""), detector=p["detector"], reader=p["reader"],
                             vehicles=p.get("vehicles"), tag=p.get("tag", "reference"))
            for name, p in cfg.items()}
