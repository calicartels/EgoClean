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

# ViT-L fpc64-256 patch layout: 8192 total = 32 temporal × 256 spatial.
# 32 temporal: 64 input frames / 2 (tubelet temporal stride) = 32.
# 256 spatial: (256/16)² = 16×16 = 256 patches per frame.
VJEPA_T_PATCHES = 32
VJEPA_S_PATCHES = 256