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

## What we tried (and why most failed)

| Approach | Result | Why |
|----------|--------|-----|
| **Optical flow periodicity** | Inverted | Head bob (2Hz) is more periodic than press cycle. Walking scored higher than work. [Fig 1](assets/flow_clip1_period.png) [Fig 2](assets/flow_clip2_period.png) |
| **RANSAC ego-motion stabilization** | Flat | Homography removed walking artifact but residual flow too noisy. Press cycle doesn't produce clean periodic signal. [Fig 3](assets/flow_clip1_stab.png) [Fig 4](assets/flow_clip2_stab.png) |
| **MediaPipe hand detection** | 37–40% detection | Blue industrial gloves. Model trained on bare skin; gloves = camouflage. |
| **V-JEPA 2 + hysteresis** | **4/4, 0 FP** | Row-mean typicality, EMA, Otsu + dual-threshold. Temporal context is the floor. [Fig 5](assets/vjepa_clip1_diag.png) [Fig 6](assets/vjepa_clip2_diag.png) [Fig 7](assets/vjepa_clip1_detect.png) [Fig 8](assets/vjepa_clip2_detect.png) |
| **DINOv2 ablation** | 138 boundaries (4 correct) | Single-frame encoder. Head turns = scene change. No temporal context. [Fig 9](assets/dino_clip1_bnd.png) [Fig 10](assets/dino_clip2_bnd.png) |

**Winner:** V-JEPA 2 embeddings → row-mean typicality → EMA → Otsu + hysteresis. $0.07 per 20-min video vs $2–7 for Gemini.

### Reading the V-JEPA outputs

**Diagnostics** ([Fig 5](assets/vjepa_clip1_diag.png) [Fig 6](assets/vjepa_clip2_diag.png)): Four panels — (1) consecutive cosine distance between embeddings (high = scene change), (2) mean-pooled similarity matrix (anomalies = dark off-diagonal), (3) autocorrelation of distance (peaks = repeating period), (4) temporal-token similarity (cycle structure).

**Detection** ([Fig 7](assets/vjepa_clip1_detect.png) [Fig 8](assets/vjepa_clip2_detect.png)): Gray = raw anomaly score (1 − typicality). Blue = EMA-smoothed. Red/orange lines = θ_on / θ_off (Otsu + hysteresis). Red shading = detected anomaly spans.

### Ground truth vs detected

Ground truth was verified by manual inspection of the video. V-JEPA detected all 4 anomalies with zero false positives:

| Clip | Ground truth | Activity | Detected |
|------|--------------|----------|----------|
| 1 | 03:28–04:22 | Material handling (not assembling) | 03:29–04:24 |
| 1 | 07:00–07:13 | Ending work | 06:56–07:11 |
| 2 | 02:07–02:54 | Not working | 02:05–03:02 |
| 2 | 16:09–17:03 | Gets up | 16:14–17:07 |

## Run

```bash
pip install -r requirements.txt
# HF_KEY in .env for gated dataset
bash run.sh
```

Runs Phase 1 (ingest + rectify) then Phase 2 (V-JEPA encode + analyze). Phase 2 needs GPU for speed; falls back to CPU/MPS if unavailable.

Phase 1 output: `data/factory_001/rectified_clip_1.mp4`, `rectified_clip_2.mp4`, `intrinsics.json`.

Phase 2 output: `data/factory_001/rectified_clip_*_emb.npy`, `*_ts.npy`, `*_diagnostics.png`.
