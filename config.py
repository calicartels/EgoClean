from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"

HF_REPO = "builddotai/Egocentric-10K"
HF_FILES = ("factory_001/workers/worker_001/factory001_worker001_part00.tar",)
FACTORY_ID = "factory_001"
EXTRACT_DIR = RAW / FACTORY_ID
TAR_PATH = RAW / "factory_001" / "workers" / "worker_001" / "factory001_worker001_part00.tar"