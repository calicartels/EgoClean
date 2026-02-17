# EgoClean

Curating egocentric factory video into training data for Vision-Language-Action (VLA) models.

I'm building a pipeline that turns raw head-mounted camera footage from factory workers into structured data that robots can learn from. The goal is 6DoF manipulation trajectories — where objects move in 3D space, with rotation — so VLAs can imitate human assembly and handling tasks.

The source is Egocentric-10K on HuggingFace: factory workers wearing fisheye cameras, doing real assembly work. The data comes as tarballs per worker (factory_001/worker_001), each containing H.265 MP4 clips (~3 min each, 30 fps) plus an intrinsics.json with fisheye calibration. For the POC we use 2 clips: factory001_worker001_00001 and factory001_worker001_00002.

The raw footage has several issues that break downstream processing. We tackle them sequentially, one phase at a time:

## Phase 1: Load and correct fisheye

**The problem:** Egocentric-10K uses wide-angle fisheye lenses (~128° FOV). Raw frames have severe barrel distortion — straight lines (table edges, shelves, tools) appear curved. Any model that assumes linear perspective (depth estimation, 3D tracking, ego-motion registration) will produce garbage if fed distorted frames. The rigid-body assumptions in later stages break when the geometry is wrong.

**What we do:** Download the tar and intrinsics from HuggingFace, extract the MP4s, rectify fisheye, and output only the corrected videos plus intrinsics. Tar, cache, and raw intermediates are deleted.

## Phase 2: Temporal structure via V-JEPA 2

**The problem:** We need to know when things happen — where action cycles start and end, which segments are repetitive, and which look anomalous. Before spending money on semantic VLMs, we want a cheap structural pass that reveals the temporal patterns in the video.

**What we do:** Slide a 64-frame V-JEPA 2 window (ViT-L, 1024-dim) across each rectified clip at 1-second stride. Mean-pool patch tokens into one embedding per window. Save embeddings as .npy. Then compute consecutive cosine distance (temporal change signal), self-similarity matrix, and autocorrelation to detect repeating structure.

## Run

```bash
# Phase 1 (local or GPU instance)
pip install -r requirements.txt
# HF_KEY in .env for gated dataset
bash run.sh

# Phase 2 (Vast.ai RTX 3090)
pip install -r requirements.txt
python VJEPA/encode.py
python VJEPA/analyze.py
```

Phase 1 output: `data/factory_001/rectified_clip_1.mp4`, `rectified_clip_2.mp4`, `intrinsics.json`.

Phase 2 output: `data/factory_001/rectified_clip_*_emb.npy`, `*_ts.npy`, `*_diagnostics.png`.
