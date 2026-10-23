# Sample media

Built-in examples for the demo page (**Try a sample…** menu).

| Files | Source | Licence |
|---|---|---|
| `photos/night_*.jpg` | NightLPBD test split — Md. Tahmid Hasan, Sk. Md. Masudul Ahsan, "NightLPBD: A YOLO-Based Single-Stage Framework for Nighttime License Plate Recognition", IEEE COMPAS 2025. https://huggingface.co/datasets/thasan3003/NightLPBD_Nighttime_License_Plates_of_Bangladesh | CC BY 4.0 |
| `photos/day_*.jpg` | Bangladesh Vehicle Number Plate Detection Dataset (Kaggle: mahedifindsdataset/bangladesh-vehicle-number-plate-detection-dataset) | CC BY 4.0 |
| `videos/*.mp4` | Generated from the photos above (zoom-in "approach", a passing vehicle, gaps); not real footage | CC BY 4.0 (derived) |

Photos were resized to at most 1600 px. Known readings with `avro-s-cctv`: all six night photos are read
correctly; `day_bus_2` is a known misread (`11-1903`, actual `11-1906`) — useful for showing **✎ Fix**.
