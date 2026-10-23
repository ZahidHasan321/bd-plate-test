"""Read a camera / video, track plates across frames, vote, and emit one event per vehicle.

Usage:
  uv run python stream.py <source> [-p avro-s-cctv] [--fps 5] [--camera gate-1] [--post URL]
    source: rtsp://user:pass@ip/stream | video.mp4 | 0 (webcam)

Each event is printed as a JSON line, appended to events.jsonl, its best plate crop saved under
events/, and (with --post) sent as JSON to a central server.
"""

import argparse
import json
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2

from engine import load_pipelines

ROOT = Path(__file__).parent


@dataclass
class Track:
    id: int
    box: list
    first: float
    last: float
    readings: list = field(default_factory=list)  # (text, bangla, valid, conf)
    best_crop: object = None
    best_conf: float = -1.0


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def vote(track: Track) -> dict:
    """Most confident-weighted reading. Readings that match the BD plate format win, but only if the same
    reading was seen at least twice; a one-off format match is often a misread."""
    seen = Counter(r[0] for r in track.readings if r[2])
    valid = [r for r in track.readings if r[2] and (seen[r[0]] >= 2 or len(track.readings) < 3)]
    pool = valid or [r for r in track.readings if r[0]]
    if not pool:
        return {"text": "", "bangla": "", "valid": False, "votes": 0}
    weight = Counter()
    for text, _, _, conf in pool:
        weight[text] += conf or 0.01
    text = weight.most_common(1)[0][0]
    bangla = next(r[1] for r in pool if r[0] == text)
    return {"text": text, "bangla": bangla, "valid": bool(valid),
            "votes": sum(r[0] == text for r in pool), "readings": len(track.readings)}


def emit(track: Track, args) -> None:
    result = vote(track)
    if track.readings and len(track.readings) >= args.min_readings and result["text"]:
        stamp = datetime.fromtimestamp(track.first).strftime("%Y%m%d_%H%M%S")
        crop = ROOT / "events" / f"{args.camera}_{stamp}_{track.id}.jpg"
        crop.parent.mkdir(exist_ok=True)
        if track.best_crop is not None:
            cv2.imwrite(str(crop), track.best_crop)
        event = {"camera": args.camera, "plate": result["text"], "bangla": result["bangla"],
                 "valid": result["valid"], "votes": result["votes"], "readings": result["readings"],
                 "first_seen": datetime.fromtimestamp(track.first).isoformat(timespec="seconds"),
                 "last_seen": datetime.fromtimestamp(track.last).isoformat(timespec="seconds"),
                 "crop": str(crop.relative_to(ROOT))}
        line = json.dumps(event, ensure_ascii=False)
        print(line, flush=True)
        with open(ROOT / "events.jsonl", "a") as f:
            f.write(line + "\n")
        if args.post:
            try:
                req = urllib.request.Request(args.post, data=line.encode(), method="POST",
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5)
            except OSError as e:
                print(f"post failed: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-p", "--pipeline", default="avro-s-cctv")
    ap.add_argument("--fps", type=float, default=5, help="frames processed per second")
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--post", help="URL to POST each event to")
    ap.add_argument("--gone", type=float, default=1.5, help="seconds unseen before a vehicle is finalised")
    ap.add_argument("--min-readings", type=int, default=2)
    args = ap.parse_args()

    pipe = load_pipelines()[args.pipeline]
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    is_file = isinstance(src, str) and not src.startswith(("rtsp", "http"))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(video_fps / args.fps)) if is_file else 1

    start = time.time()
    tracks: list[Track] = []
    next_id, frame_no, processed, last_run, busy = 1, 0, 0, 0.0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        # Video files: use their own clock and every Nth frame. Live: process at most --fps per second.
        now = start + frame_no / video_fps if is_file else time.time()
        if is_file and frame_no % step:
            continue
        if not is_file and now - last_run < 1 / args.fps:
            continue
        last_run = now
        processed += 1
        t0 = time.perf_counter()
        res = pipe.run(frame)
        busy += time.perf_counter() - t0

        for p in res["plates"]:
            match = max(tracks, key=lambda t: iou(t.box, p["box"]), default=None)
            if match is None or iou(match.box, p["box"]) < 0.2:
                match = Track(next_id, p["box"], now, now)
                tracks.append(match)
                next_id += 1
            match.box, match.last = p["box"], now
            match.readings.append((p["text"], p["bangla"], p["valid"], p["conf"]))
            score = (p["conf"] or 0) + p["det_conf"]
            if score > match.best_conf:
                match.best_conf, match.best_crop = score, p["crop"]

        for t in [t for t in tracks if now - t.last > args.gone]:
            emit(t, args)
            tracks.remove(t)

    for t in tracks:
        emit(t, args)
    print(f"done: {frame_no} frames read, {processed} processed, "
          f"{busy / max(processed, 1) * 1000:.0f} ms per processed frame", flush=True)


if __name__ == "__main__":
    main()
