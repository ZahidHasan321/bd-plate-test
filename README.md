# plate-test

Test Bangladeshi licence-plate recognition models with your phone camera. The models run on your computer's
CPU (no GPU needed); the phone just shows the camera and the results.

## Easiest: let Claude Code do it

Open a terminal in this folder, start `claude`, and type:

> Set up and start this project so I can test it on my phone.

Claude Code reads `CLAUDE.md` and runs the steps below for you.

## By hand (Linux / macOS / Windows WSL)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
   `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Download and convert the models (about 400 MB, one time):
   `./fetch_weights.sh`
3. Start it — pick one:

   | Command | When | Phone opens |
   |---|---|---|
   | `./run.sh tunnel` | **Easiest.** Phone on any network (mobile data is fine) | the `https://….trycloudflare.com` link it prints |
   | `./run.sh` | Phone on the **same Wi-Fi** as the computer | `https://<computer-ip>:8000` — tap *Advanced → Proceed* on the certificate warning once |

   The tunnel link is public while it runs (anyone who has it can use the page); stop it with Ctrl+C.

4. Pick a model (the dropdown groups them: **★ Recommended** `avro-s-cctv`, Faster, Other, Reference, Baseline —
   the line under it shows each model's test numbers), then:
   - **Sample photo…** — built-in night/day photos (see `samples/README.md`).
   - **Camera** — phone (or laptop) camera, live.
   - **Video** — pick a clip on this device (laptop or phone gallery). It is uploaded once and analysed on the
     computer (~10 frames per second of video); playback pauses briefly whenever it gets ahead of the analysis,
     so no part is skipped. **1× / 0.5× / 0.25×** changes the playback speed.
   - **Photos** — one or several pictures (on a phone: camera or gallery); several give a results table.
     **Compare all** runs the last frame through every pipeline.

   The **Vehicles** list shows **one row per vehicle**: plate crop, English + Bangla text, and a badge —
   *read*, *needs review* (seen clearly but not readable), or *confirmed*. A car that waits, is hidden for a
   moment, or comes back within 2 minutes stays one row (same 6 digits, or one digit off), and its reading
   improves while it is in view. **✓ Correct** / **✎ Fix** confirm or correct a row; **Clear** empties the list.

Green box = reading matches the BD plate format (district + letter + 6 digits); orange = it doesn't.

## Pipelines

Defined in `pipelines.yaml` (plate detector → character reader). Best so far for cheap edge devices: **`avro-s-cctv`** (or `avro-s-cctv-fast` for ~2× faster reading).

| Pipeline | Detector → reader | Notes |
|---|---|---|
| **`avro-s-cctv`** | night models + detector trained on toll-booth CCTV | finds **99%** of plates on held-out CCTV frames (was 88%); 72% plates fully correct on NightLPBD |
| **`avro-s-cctv-fast`** | same, 320px reader | same detection, reader ~2× faster |
| `avro-s-night` | YOLOv8n → YOLOv8s, fine-tuned | **74%** plates fully correct on NightLPBD test (was 62%) |
| `avro-s-night-fast` | same, reader trained at 320px | **70%**, reader ~2× faster |
| `avro-s` | YOLOv8n → YOLOv8s (original Avro) | 62%; runs on Jetson & RK3588 |
| `avro-s-fast` | original reader run at 320px | 44% — use `avro-s-night-fast` instead |
| `avro-n` | YOLOv8n → YOLOv8n | cheapest |
| `oim-avro-s` | YOLOv9-t (MIT) → YOLOv8s | |
| `rashed` | YOLOv8n → YOLO11n | |
| `blpd-lite` / `blpd` | YOLOv8n or v12m → YOLOv12n | YOLOv12 on RK3588 unverified |
| `maidul` | YOLOv8m → YOLOv8m | heavy accuracy reference |
| `dhaka-cctv` / `dhaka-direct` | (vehicles →) YOLO26s Dhaka-CCTV plate detector → YOLOv8s | trained on daytime high-mounted CCTV |
| `global` | fast-alpr global models | Latin plates only — baseline |

## Real cameras / video: one result per vehicle

```
uv run python stream.py rtsp://user:pass@CAMERA_IP/stream --camera gate-1   # IP camera
uv run python stream.py video.mp4                                            # video file
uv run python stream.py 0                                                    # webcam
```
Same vehicle logic as the web page (`tracker.py`): ~5 frames/s, one record per vehicle in `events.db`
(updated as readings improve), best crop in `events/<camera>/`, a log line in `events.jsonl`. With
`--post http://server/api` every new/updated record is sent with a stable `id` + `version` and retried until the
server answers 2xx (upsert by `id` on the server). Live cameras reconnect automatically, including when the
stream freezes. `--ignore x1,y1,x2,y2` skips a region (e.g. the camera's timestamp overlay); see `--help`.

## Benchmark on a folder of images

```
uv run python bench.py <image-folder>                       # readings + timings
uv run python bench.py <folder> --truth gt.csv --csv out.csv # + accuracy
```
`gt.csv` columns: `image,area,letter,digits` (e.g. `car1.jpg,DHAKA METRO,GA,123456`).
Runs unchanged on an edge board with onnxruntime.

## Retraining

`train/train_kaggle.py` fine-tunes `avro-s` on NightLPBD (night plates) + LPDB-A on a free Kaggle GPU:
`cd train && uvx kaggle kernels push -p .` (edit the `id` in `kernel-metadata.json` to your Kaggle username).

Also `train/detect_cctv/`: further trains the plate detector on a private CCTV dataset (needs that dataset on your
Kaggle account; edit `kernel-metadata.json`).

## Add a model

1. Put the `.pt` in `weights/raw/<repo>/<name>.pt`.
2. `uv run --group export python export.py` → `weights/onnx/<repo>_<name>.onnx`.
3. Add an entry to `pipelines.yaml` and restart.

Every reading is logged to `results.csv`, plate crops to `captures/<pipeline>/`.

## Licences

The models are downloaded from their authors' repos by `fetch_weights.sh`
(AvroHabib, BLPD, MaidulHasan, rashedulalbab253, zarifbinhasnat, Ultralytics). Several have no licence and all
were trained with Ultralytics (AGPL-3.0): **internal testing only**, not for commercial use.
