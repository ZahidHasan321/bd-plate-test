"""Turn per-frame plate readings into one event per vehicle (used by stream.py and the web app).

- Plates are followed across frames (IoU), readings are voted (confidence-weighted, large/sharp crops first).
- A vehicle is reported as soon as it is confidently read (also while standing still) and updated as better
  readings arrive; a format-valid reading counts if seen twice or once with high confidence (fast vehicles).
- Same plate on the same camera within `dedup` seconds (digits equal or one digit off) updates the earlier
  event instead of creating a new one (broken tracks, waiting vehicles, trucks with two plates, misreads).
- Plates seen several times but never readable become status "needs_review"; brief junk detections are dropped.
Events live in events.db (SQLite, WAL), crops under events/<camera>/, a log line in events.jsonl.
"""

import json
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).parent
DB = ROOT / "events.db"


@dataclass
class Settings:
    gone: float = 1.5          # seconds unseen before a vehicle's track closes
    early: float = 2.0         # report a confident read after this many seconds on screen
    dedup: float = 120         # same plate on this camera within N s = same vehicle
    single_conf: float = 0.85  # trust a single format-valid reading above this confidence
    min_plate_px: int = 40     # narrower plate crops are low quality
    min_sharpness: float = 20  # Laplacian variance below this = blurry
    review_after: int = 3      # unreadable after N readings -> needs_review
    ignore: tuple = ()         # regions (x1, y1, x2, y2 as 0-1 fractions) to ignore, e.g. timestamp overlay


def connect(path=DB):
    db = sqlite3.connect(path, timeout=30, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY, camera TEXT, plate TEXT, bangla TEXT, digits TEXT, valid INTEGER,
        status TEXT, votes INTEGER, readings INTEGER, first_seen REAL, last_seen REAL, crop TEXT,
        version INTEGER DEFAULT 1, sent_version INTEGER DEFAULT 0,
        reviewed_plate TEXT, reviewed INTEGER DEFAULT 0)""")
    db.commit()
    return db


@dataclass
class Track:
    id: int
    box: list
    first: float
    last: float
    readings: list = field(default_factory=list)  # dicts: text, bangla, digits, valid, conf, good
    best_crop: object = None
    best_score: float = -1.0
    event_id: str | None = None
    recorded_at: int = 0  # number of readings when last recorded
    recorded_text: str = ""


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def near(a, b):
    """Same plate number: 6 digits equal or one digit different (a misread)."""
    return len(a) == len(b) == 6 and sum(x != y for x, y in zip(a, b)) <= 1


def vote(track, s: Settings) -> dict:
    readings = [r for r in track.readings if r["good"]] or track.readings
    seen = Counter(r["text"] for r in readings if r["valid"])
    trusted = [r for r in readings if r["valid"] and (seen[r["text"]] >= 2 or (r["conf"] or 0) >= s.single_conf)]
    pool = trusted or [r for r in readings if r["text"]]
    if not pool:
        return {"text": "", "confident": False, "readings": len(track.readings)}
    weight = Counter()
    for r in pool:
        weight[r["text"]] += r["conf"] or 0.01
    text = weight.most_common(1)[0][0]
    best = next(r for r in pool if r["text"] == text)
    return {"text": text, "bangla": best["bangla"], "digits": best["digits"], "valid": bool(trusted),
            "confident": bool(trusted), "votes": sum(r["text"] == text for r in pool), "readings": len(track.readings)}


def readable_digits(track):
    """A real plate that is hard to read still yields some digits; boxes on signs or plate fragments rarely do."""
    return max((len(r["digits"]) for r in track.readings), default=0) >= 3


class PlateTracker:
    def __init__(self, camera: str, settings: Settings | None = None, db=None, log=print):
        self.camera, self.s = camera, settings or Settings()
        self.db, self.log = db or connect(), log
        self.tracks: list[Track] = []
        self.next_id = 1
        (ROOT / "events" / camera).mkdir(parents=True, exist_ok=True)

    def update(self, now: float, frame_shape, plates: list) -> None:
        """Feed one processed frame: `plates` as returned by Pipeline.run (with crops)."""
        s, (h, w) = self.s, frame_shape[:2]
        # Drop boxes lying inside a larger plate box (e.g. a second box around just the top line).
        def inside(a, b):
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            return ix >= 0.7 * (a[2] - a[0]) * (a[3] - a[1])
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        plates = [p for p in plates if not any(q is not p and area(q["box"]) > area(p["box"]) and inside(p["box"], q["box"])
                                               for q in plates)]
        for p in plates:
            cx, cy = (p["box"][0] + p["box"][2]) / 2 / w, (p["box"][1] + p["box"][3]) / 2 / h
            if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in s.ignore):
                continue
            t = max(self.tracks, key=lambda t: iou(t.box, p["box"]), default=None)
            if t is None or iou(t.box, p["box"]) < 0.2:
                t = Track(self.next_id, p["box"], now, now)
                self.tracks.append(t)
                self.next_id += 1
            t.box, t.last = p["box"], now
            crop = p["crop"]
            good = crop.shape[1] >= s.min_plate_px and \
                cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() >= s.min_sharpness
            t.readings.append({"text": p["text"], "bangla": p["bangla"], "digits": p.get("digits", ""),
                               "valid": p["valid"], "conf": p["conf"], "good": good})
            score = (p["conf"] or 0) + p["det_conf"] + (1 if good else 0)
            if score > t.best_score:
                t.best_score, t.best_crop = score, crop

        for t in list(self.tracks):
            ended = now - t.last > s.gone
            result = vote(t, s)
            if result["confident"] and (ended or now - t.first >= s.early or t.event_id):
                if ended or t.event_id is None or result["text"] != t.recorded_text or len(t.readings) - t.recorded_at >= 25:
                    self.record(t, result, "ok")  # early report, periodic refresh while visible, final on exit
            elif ended and len(t.readings) >= s.review_after and readable_digits(t):
                self.record(t, result, "needs_review")
            if ended:
                self.tracks.remove(t)  # few readings and never confident: noise, dropped

    def close_all(self) -> None:
        for t in self.tracks:
            result = vote(t, self.s)
            if result["confident"]:
                self.record(t, result, "ok")
            elif len(t.readings) >= self.s.review_after and readable_digits(t):
                self.record(t, result, "needs_review")
        self.tracks = []

    def record(self, track, result, status):
        """Insert a new event, or update the matching one (same track, or same plate within dedup)."""
        db, now = self.db, track.last
        event_id = track.event_id
        if event_id is None and result.get("digits"):
            for r in db.execute("SELECT id, digits FROM events WHERE camera=? AND last_seen>=?",
                                (self.camera, now - self.s.dedup)).fetchall():
                if near(r["digits"], result["digits"]):
                    event_id = r["id"]
                    break
        crop = ROOT / "events" / self.camera / f"{datetime.fromtimestamp(track.first):%Y%m%d_%H%M%S}_{track.id}.jpg"
        if track.best_crop is not None:
            cv2.imwrite(str(crop), track.best_crop)
        fields = dict(camera=self.camera, plate=result["text"], bangla=result.get("bangla", ""),
                      digits=result.get("digits", ""), valid=int(result.get("valid", False)), status=status,
                      votes=result.get("votes", 0), readings=result.get("readings", 0),
                      first_seen=track.first, last_seen=now, crop=str(crop.relative_to(ROOT)))
        old = db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone() if event_id else None
        if old is None:
            event_id = event_id or uuid.uuid4().hex
            db.execute(f"INSERT INTO events (id, {', '.join(fields)}) VALUES (?{', ?' * len(fields)})",
                       (event_id, *fields.values()))
            action = "new"
        else:
            # Keep the better reading (confident beats unsure, then more votes); always extend the time span.
            if (fields["valid"], fields["votes"]) <= (old["valid"], old["votes"]):
                fields = {}
            if old["status"] == "ok":
                fields.pop("status", None)
            fields["first_seen"] = min(old["first_seen"], track.first)
            fields["last_seen"] = max(old["last_seen"], now)
            db.execute(f"UPDATE events SET {', '.join(f'{k}=?' for k in fields)}, version=version+1 WHERE id=?",
                       (*fields.values(), event_id))
            action = "update"
        db.commit()
        track.event_id, track.recorded_at, track.recorded_text = event_id, len(track.readings), result["text"]
        row = dict(db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone())
        line = json.dumps({"action": action, **{k: row[k] for k in ("id", "camera", "plate", "bangla", "status",
                                                                     "valid", "votes", "readings", "version", "crop")},
                           "first_seen": datetime.fromtimestamp(row["first_seen"]).isoformat(timespec="seconds"),
                           "last_seen": datetime.fromtimestamp(row["last_seen"]).isoformat(timespec="seconds")},
                          ensure_ascii=False)
        self.log(line)
        with open(ROOT / "events.jsonl", "a") as f:
            f.write(line + "\n")
