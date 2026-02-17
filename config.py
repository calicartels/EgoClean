from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "factory_001"

HF_REPO = "builddotai/Egocentric-10K"
HF_FILES = (
    "factory_001/workers/worker_001/factory001_worker001_part00.tar",
    "factory_001/workers/worker_001/intrinsics.json",
)
FACTORY_ID = "factory_001"
TAR_PATH = DATA / "factory_001" / "workers" / "worker_001" / "factory001_worker001_part00.tar"
EXTRACT_DIR = DATA / "_tmp"
INTRINSICS_PATH = EXTRACT_DIR / "intrinsics.json"
FISHEYE_BALANCE = 0
EXPECTED_CLIPS = 2
VIDEO_CRF = 28

# V-JEPA 2 encoding (Phase 2)

VJEPA_REPO = "facebook/vjepa2-vitl-fpc64-256"
VJEPA_WINDOW = 64
VJEPA_STRIDE_SEC = 1.0
VJEPA_RESIZE = 256

# ViT-L fpc64-256 patch layout: 8192 total = 32 temporal x 256 spatial.
VJEPA_T_PATCHES = 32
VJEPA_S_PATCHES = 256

# Anomaly detection

# Choice: 10th percentile of row-mean similarity. Flags windows whose average
# similarity to all others is in the bottom 10%. Nonparametric — adapts per clip.
# Alternative: mean-2*std on consecutive similarity (failed — signal too noisy,
# threshold too low, nothing flagged). Or fixed threshold (doesn't generalize).
ANOMALY_PERCENTILE = 10

# Choice: merge anomalies within 5s. A brief return to "normal" mid-transit
# (e.g. worker glances at workstation while walking) shouldn't split one event.
# Alternative: 2s (tighter, more fragments) or 10s (risks merging separate events).
ANOMALY_MIN_GAP = 5.0

# Choice: discard anomalies shorter than 3s. Head turns and glances cause
# 1-2s similarity dips that aren't real departures from work.
# Alternative: 1s (catches more, more false positives) or 5s (misses brief events).
ANOMALY_MIN_DUR = 3.0