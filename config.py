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
