# EgoClean

Curating egocentric factory video into training data for Vision-Language-Action (VLA) models.

I'm building a pipeline that turns raw head-mounted camera footage from factory workers into structured data that robots can learn from. The goal is 6DoF manipulation trajectories — where objects move in 3D space, with rotation — so VLAs can imitate human assembly and handling tasks.

The source is Egocentric-10K on HuggingFace: factory workers wearing fisheye cameras, doing real assembly work. The data comes as tarballs per worker (factory_001/worker_001), each containing H.265 MP4 clips (~3 min each, 30 fps) plus an intrinsics.json with fisheye calibration. For the POC we use 2 clips: factory001_worker001_00001 and factory001_worker001_00002.

The raw footage has several issues that break downstream processing. We tackle them sequentially, one phase at a time:

## Phase 1: Load and correct fisheye

**The problem:** Egocentric-10K uses wide-angle fisheye lenses (~128° FOV). Raw frames have severe barrel distortion — straight lines (table edges, shelves, tools) appear curved. Any model that assumes linear perspective (depth estimation, 3D tracking, ego-motion registration) will produce garbage if fed distorted frames. The rigid-body assumptions in later stages break when the geometry is wrong.

**What we do:** Download the tar and intrinsics from HuggingFace, extract the MP4s, rectify fisheye, and output only the corrected videos plus intrinsics. Tar, cache, and raw intermediates are deleted.

## Run

```bash
pip install -r requirements.txt
# HF_KEY in .env for gated dataset
bash run.sh
```

Output: `data/factory_001/data_point_001.mp4`, `data_point_002.mp4`, `intrinsics.json`.
