# triage

Stage 1: Grounding DINO per-frame detection, IoU hand-object flagging, segment merging.

Run: `python -m triage.detect` then `python -m triage.merge`, or `bash run.sh triage`.

Output: `data/triage/{clip}_detections.json`, `data/triage/{clip}_manifest.json`.
