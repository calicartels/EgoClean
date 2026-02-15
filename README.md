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

## Phase 2: Semantic annotation with Qwen2.5-VL-7B

**The problem:** The footage is a mix of active manipulation (hands on objects, assembly, picking, placing) and non-manipulation (idle hands, walking between workstations, resting). We need to know *which* frames matter and *what* the worker is doing. The traditional approach uses Grounding DINO for hand/object detection, then merges consecutive active frames into segments — two models, two steps, more moving parts.

**What we do:** Skip triage. Feed the entire clip as a video (rectified frames at 1 FPS, ~180 frames) to Qwen2.5-VL-7B in one call. The VLM does temporal action localization: it returns a list of action segments with start/end timestamps, natural boundaries (no arbitrary 5-second cuts splitting actions in half). Each segment has an `objects` array — empty for idle, one entry for single-object manipulation, multiple entries when the worker is assembling part A into part B. Each object has optional `object_name`, `bbox`, and `rigid` flag. ECoT reasoning (scene, subtask, motion, prediction) per segment. Output goes to `data/annotate/{clip}_annotations.json`. Raw model response saved to `{clip}_raw_response.txt` so if JSON parsing fails we see exactly what the model returned.

---

I'm targeting a single rented RTX 3090 on Vast.ai — 24 GB VRAM, ~$0.09–0.20/hr. Budget matters. The pipeline runs phases sequentially, loads one model at a time, and keeps costs under a few dollars for the POC.

**TL;DR**

- **Phase 1:** Download, extract, rectify fisheye in-memory. Fixes geometry so later stages don't drift.
- **Phase 2:** Qwen2.5-VL-7B over whole clip as video (~13k input tokens); temporal action localization with `objects` array (0..N per segment). One call per clip.
- **Output:** `data/annotate/{clip}_annotations.json` — action segments with natural boundaries, `objects` array for multi-object support
- **Runtime:** ~1–3 min per clip on 3090 after model download (~14 GB for 7B, cached)

### Qwen2.5-VL API: input and output

**Input format:** Chat-style messages with mixed content. We pass a video (list of PIL images) and a text prompt:

```python
content = [
    {"type": "video", "video": [pil_frame_0, pil_frame_1, ...], "sample_fps": 1.0},
    {"type": "text", "text": PROMPT},
]
messages = [{"role": "user", "content": content}]
```

`qwen_vl_utils.process_vision_info(messages, return_video_kwargs=True)` extracts image/video tensors and video kwargs. The processor consumes these plus the chat template text. At 456×256 frames, 1 FPS, ~180 frames → ~13k visual tokens. Context window is 32k; we use ~42%.

**Output format:** Raw autoregressive text. We ask for a JSON array; the model returns a string that we parse. Each segment:

```json
{
  "action_id": 1,
  "t_start_s": 5,
  "t_end_s": 12,
  "active": true,
  "objects": [
    {"object_name": "screwdriver", "bbox": [x1, y1, x2, y2], "rigid": true}
  ],
  "ecot": {
    "scene": "...",
    "subtask": "...",
    "motion": "...",
    "prediction": "..."
  }
}
```

Idle segments have `active: false`, `objects: []`, `ecot: null`. Raw response is saved to `{clip}_raw_response.txt` before parsing.

## Quick start

```bash
pip install -r requirements.txt
# Set HF_KEY for HuggingFace. Use export so child processes inherit it:
export HF_KEY=your_token
bash run.sh all
# Or inline: HF_KEY=your_token bash run.sh all
```

Or step by step: Phase 1 (`download` → `extract` → `check-rectify`) then Phase 2 (`annotate`).
