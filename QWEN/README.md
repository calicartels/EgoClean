Qwen2.5-VL labeling for egocentric factory video. Run from EgoClean root.

`label.py` — GPU. Slides 30s windows at 1fps, asks VLM for work/off-task transitions.
Outputs per-clip `*_raw.json` (model responses) and `*_labels.json` (per-second labels).
