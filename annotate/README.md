# annotate

Stage 2: Qwen2.5-VL per-window ECoT annotation.

Run: `python -m annotate.annotate`, or `bash run.sh annotate`.

Output: `data/annotate/{clip}_annotations.json` with per-window structured JSON.
Each window is either `active: true` (with object name, bbox, ECoT reasoning) or `active: false` (idle/transit).
