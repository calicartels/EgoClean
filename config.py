from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"

HF_REPO = "builddotai/Egocentric-100K"
HF_FILES = ("factory001/worker001/part000.tar", "factory001/worker001/intrinsics.json")

POC_CLIPS = ("factory_001_worker_001_0000", "factory_001_worker_001_0001")
TAR_DIR = RAW / "factory001" / "worker001"
INTRINSICS_PATH = TAR_DIR / "intrinsics.json"
DEBUG_DIR = DATA / "debug"

# --- Triage (Stage 1) ---

TRIAGE_DIR = DATA / "triage"

TRIAGE_FPS = 1
TRIAGE_PROMPT = "human hand . target object ."

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

IOU_THRESHOLD = 0.05

MERGE_GAP_S = 2.0
MIN_DURATION_S = 1.5
