"""Read a camera / video, track plates across frames, vote, and record one event per vehicle.

Usage:
  uv run python stream.py <source> [-p avro-s-cctv] [--camera gate-1] [--post URL]
    source: rtsp://user:pass@ip/stream | video.mp4 | 0 (webcam)

Events go to events.db (SQLite, survives reboots/power cuts), are printed as JSON lines and appended to
events.jsonl, with the best plate crop under events/<camera>/. With --post, a background sender delivers
every new or updated event to a central server, retrying until it answers 2xx. Each event carries a stable
`id` and a `version`: the server should upsert by id (repeats are harmless).

Live-camera handling: reconnect with backoff, reconnect on a frozen stream, always use the newest frame,
timestamps taken when the frame is grabbed. Event logic: a vehicle is reported once it is confidently read
(early, even if it is still standing still) and the record is updated as better readings arrive; the same
plate on the same camera within --dedup seconds (digits equal or one digit off) updates the earlier event
instead of creating a new one; plates that are seen clearly but cannot be read become status "needs_review".
"""

import argparse
import json
import threading
import time
import urllib.request
from datetime import datetime

import cv2

from engine import load_pipelines
from tracker import ROOT, PlateTracker, Settings, connect

# ---------------------------------------------------------------- frame sources
class FileSource:
    """Video file: every Nth frame, timestamps from the video's own clock."""

    live = False

    def __init__(self, path, fps):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.step = max(1, round(self.video_fps / fps))
        self.start, self.n = time.time(), 0

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            self.n += 1
            if self.n % self.step == 0:
                yield self.start + self.n / self.video_fps, frame

    def status(self):
        return {"connected": True}


class LiveSource:
    """RTSP / webcam: a reader thread keeps only the newest frame (no queued lag), reconnects with backoff
    when the stream fails, and also reconnects when frames stop changing (frozen stream that still 'works')."""

    live = True

    def __init__(self, src, fps, stale=5.0):
        self.src, self.fps, self.stale = src, fps, stale
        self.lock = threading.Lock()
        self.frame, self.stamp, self.seq = None, 0.0, 0
        self.connected, self.reconnects, self.last_change = False, 0, time.time()
        threading.Thread(target=self._reader, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self):
        backoff = 1.0
        while True:
            cap = self._open()
            if not cap.isOpened():
                self.connected = False
                print(f"[source] cannot open stream, retry in {backoff:.0f}s", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            self.connected, backoff, last_sig = True, 1.0, None
            self.last_change = time.time()
            print("[source] connected", flush=True)
            while True:
                ok, frame = cap.read()
                now = time.time()
                if not ok:
                    print("[source] stream lost", flush=True)
                    break
                sig = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA).tobytes()
                if sig != last_sig:
                    last_sig, self.last_change = sig, now
                elif now - self.last_change > self.stale:
                    print(f"[source] frozen for {self.stale:.0f}s, reconnecting", flush=True)
                    break
                with self.lock:
                    self.frame, self.stamp, self.seq = frame, now, self.seq + 1
            cap.release()
            self.connected = False
            self.reconnects += 1
            time.sleep(1)

    def frames(self):
        seen = 0
        while True:
            time.sleep(1 / self.fps)
            with self.lock:
                if self.seq == seen or self.frame is None:
                    continue
                seen, frame, stamp = self.seq, self.frame, self.stamp
            yield stamp, frame

    def status(self):
        return {"connected": self.connected, "reconnects": self.reconnects,
                "last_frame_age": round(time.time() - self.stamp, 1) if self.stamp else None}


# ---------------------------------------------------------------- sender
def post_json(url, row):
    body = json.dumps(row, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return 200 <= r.status < 300


def sender(url):
    """Deliver new/updated events until the server accepts them; survives restarts (state is in SQLite)."""
    db, backoff = connect(), 2.0
    while True:
        rows = db.execute("SELECT * FROM events WHERE sent_version < version ORDER BY last_seen LIMIT 50").fetchall()
        if not rows:
            time.sleep(2)
            continue
        try:
            for r in rows:
                row = {k: r[k] for k in r.keys() if k != "sent_version"}
                if post_json(url, row):
                    db.execute("UPDATE events SET sent_version=? WHERE id=?", (r["version"], r["id"]))
                    db.commit()
            backoff = 2.0
        except OSError as e:
            print(f"[sender] server unreachable ({e}); {len(rows)} event(s) queued, retry in {backoff:.0f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-p", "--pipeline", default="avro-s-cctv")
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--fps", type=float, default=5, help="frames processed per second")
    ap.add_argument("--post", help="URL to POST events to (retried until accepted)")
    ap.add_argument("--gone", type=float, default=1.5, help="seconds unseen before a vehicle's track closes")
    ap.add_argument("--early", type=float, default=2.0, help="report a confident read after this many seconds")
    ap.add_argument("--dedup", type=float, default=120, help="same plate on this camera within N s = same vehicle")
    ap.add_argument("--single-conf", type=float, default=0.85, help="trust a single valid reading above this")
    ap.add_argument("--min-plate-px", type=int, default=40, help="plate crops narrower than this are low quality")
    ap.add_argument("--min-sharpness", type=float, default=20, help="Laplacian variance below this = blurry")
    ap.add_argument("--review-after", type=int, default=3, help="unreadable after N readings -> needs_review")
    ap.add_argument("--ignore", action="append", default=[], metavar="x1,y1,x2,y2",
                    help="ignore plates whose centre is in this region (fractions 0-1), e.g. timestamp overlay")
    args = ap.parse_args()

    if args.post:
        threading.Thread(target=sender, args=(args.post,), daemon=True).start()
    tracker = PlateTracker(args.camera, Settings(
        gone=args.gone, early=args.early, dedup=args.dedup, single_conf=args.single_conf,
        min_plate_px=args.min_plate_px, min_sharpness=args.min_sharpness, review_after=args.review_after,
        ignore=tuple(tuple(map(float, r.split(","))) for r in args.ignore)), log=lambda l: print(l, flush=True))
    pipe = load_pipelines()[args.pipeline]
    is_live = args.source.isdigit() or args.source.startswith(("rtsp", "http"))
    src = LiveSource(int(args.source) if args.source.isdigit() else args.source, args.fps) if is_live \
        else FileSource(args.source, args.fps)
    heartbeat = ROOT / "events" / f"heartbeat_{args.camera}.json"

    processed, busy, last_beat = 0, 0.0, 0.0
    for now, frame in src.frames():
        processed += 1
        t0 = time.perf_counter()
        res = pipe.run(frame)
        busy += time.perf_counter() - t0
        tracker.update(now, frame.shape, res["plates"])
        if time.time() - last_beat > 10:
            last_beat = time.time()
            heartbeat.write_text(json.dumps({"camera": args.camera, "time": datetime.now().isoformat(timespec="seconds"),
                                             "processed": processed, "open_tracks": len(tracker.tracks),
                                             "ms_per_frame": round(busy / max(processed, 1) * 1000), **src.status()}))
    tracker.close_all()  # end of a video file
    print(f"done: {processed} frames processed, {busy / max(processed, 1) * 1000:.0f} ms per frame", flush=True)
    if args.post:
        time.sleep(3)  # give the sender a moment to flush before a file run exits

if __name__ == "__main__":
    main()
