# plate-test — instructions for Claude Code

Phone-camera test bench for Bangladeshi licence-plate recognition. Python, FastAPI, onnxruntime (CPU).

## When the user asks to set up / run / test on their phone

1. Check `uv` is installed (`uv --version`); if not, install it:
   `curl -LsSf https://astral.sh/uv/install.sh | sh` (then use `~/.local/bin/uv` if not on PATH).
2. If `weights/onnx/` is missing or empty, run `./fetch_weights.sh` (downloads ~400 MB, installs CPU PyTorch
   for a one-time ONNX export; takes several minutes — run it in the background and report progress).
3. Ask which way the phone connects, defaulting to the tunnel:
   - `./run.sh tunnel` — works on any network. Run it in the background, wait for the
     `https://<random>.trycloudflare.com` line in its output, and give that link to the user. Tell them the link
     is public while it runs.
   - `./run.sh` — phone on the same Wi-Fi; give the printed `https://<ip>:8000` and tell them to tap
     *Advanced → Proceed* on the certificate warning once. If the phone can't connect, check the computer's
     firewall allows port 8000.
4. Tell the user: choose a pipeline in the dropdown (`avro-s-night` is the best cheap one; `avro-s-night-fast` is faster), press **Start live** or
   **Photo**, and **Compare all** to compare every pipeline on the same frame.

## Layout

- `engine.py` — YOLO ONNX runner, BD plate grammar (`assemble`), pipelines from `pipelines.yaml`.
- `app.py` + `templates/index.html` — web app; `/predict?pipeline=`, `/compare`.
- `bench.py` — batch evaluation (`--truth gt.csv` for accuracy).
- `stream.py` — RTSP/video/webcam → tracking + voting → one JSON event per vehicle (`events.jsonl`, `--post URL`).
- `export.py` — `.pt` → ONNX (needs `uv run --group export`).
- `train/` — Kaggle fine-tuning job.

Do not commit `weights/`, `captures/`, `results.csv`, `events/`, `events.jsonl`, `cert.pem`, `key.pem`.
