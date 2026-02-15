# EgoClean

Curating egocentric factory video into training data for Vision-Language-Action (VLA) models.

I'm building a pipeline that turns raw head-mounted camera footage from factory workers into structured data that robots can learn from. The goal is 6DoF manipulation trajectories — where objects move in 3D space, with rotation — so VLAs can imitate human assembly and handling tasks.

The source is **Egocentric-100K** on HuggingFace: factory workers wearing fisheye cameras, doing real assembly work. The data comes as tarballs per worker (`factory_001/worker_001`), each containing H.265 MP4 clips (~3 min each, 30 fps) plus an `intrinsics.json` with fisheye calibration. For the POC we use 2 clips: `factory_001_worker_001_0000` and `factory_001_worker_001_0001`.

The raw footage has several issues that break downstream processing. We tackle them **sequentially**, one phase at a time:

---

## Phase 1: Load and correct fisheye

**The problem:** Egocentric-100K uses wide-angle fisheye lenses (~128° FOV). Raw frames have severe barrel distortion — straight lines (table edges, shelves, tools) appear curved. Any model that assumes linear perspective (depth estimation, 3D tracking, ego-motion registration) will produce garbage if fed distorted frames. The rigid-body assumptions in later stages break when the geometry is wrong.

**What we do:** Download the tar and intrinsics from HuggingFace, extract the MP4s, and rectify every frame in-memory using OpenCV's fisheye model (K, D from `intrinsics.json`). We never save undistorted video to disk — that would blow up storage. All downstream stages consume rectified frames through a single `iter_frames()` generator. `check-rectify` dumps a raw vs rectified pair to `data/debug/` so you can visually confirm the fix.

---

## Phase 2: Semantic annotation with Qwen2.5-VL

**The problem:** The footage is a mix of active manipulation (hands on objects, assembly, picking, placing) and non-manipulation (idle hands, walking between workstations, resting). We need to know *which* frames matter and *what* the worker is doing. The traditional approach uses Grounding DINO for hand/object detection, then merges consecutive active frames into segments — two models, two steps, more moving parts.

**What we do:** Skip triage. Feed 5-second windows (5 frames at 1 FPS) directly to Qwen2.5-VL-7B. The VLM decides: active or idle? If active, it outputs object name, bounding box, rigid/non-rigid flag, and ECoT reasoning (scene, subtask, motion, prediction). One model does the job. Output goes to `data/annotate/{clip}_annotations.json` — structured JSON per window, ready for downstream masking and 3D tracking.

---

I'm targeting a single rented RTX 3090 on Vast.ai — 24 GB VRAM, ~$0.09–0.20/hr. Budget matters. The pipeline runs phases sequentially, loads one model at a time, and keeps costs under a few dollars for the POC.

**TL;DR**

- **Phase 1:** Download, extract, rectify fisheye in-memory. Fixes geometry so later stages don't drift.
- **Phase 2:** Qwen2.5-VL-7B over 5-frame windows at 1 FPS; outputs active/idle, object name, bbox, ECoT. Replaces triage + merge with one VLM.
- **Output:** `data/annotate/{clip}_annotations.json` — ~36 windows per clip, 72 total for 2 POC clips
- **Runtime:** ~40–60 min on 3090 after model download (~14 GB for 7B, cached)
- **Planned:** SAM 3 masking, SpaTracker 3D tracking, ego-motion cancellation, 6DoF extraction, BGTS filter — see `plan.md` for the full roadmap
- **Caveat:** If Qwen returns malformed JSON (wraps it in extra text), the script crashes on purpose so we see the bad output and fix the prompt instead of silently dropping it

## Quick start

```bash
pip install -r requirements.txt
# Set HF_KEY in .env for HuggingFace download
bash run.sh all
```

Or step by step: Phase 1 (`download` → `extract` → `check-rectify`) then Phase 2 (`annotate`).
