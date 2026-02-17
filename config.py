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

# Anomaly detection — hysteresis (LAPS-style). Signal: 1 - row_mean_similarity.
DETECT_EMA_ALPHA = 0.15
DETECT_THETA_ON = None  # None = Otsu auto, else float override
DETECT_HYSTERESIS_RATIO = 0.5  # theta_off = median + ratio * (theta_on - median)
DETECT_DEBOUNCE_ON = 3
DETECT_DEBOUNCE_OFF = 3
DETECT_MIN_DUR = 3.0