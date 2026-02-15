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

# --- Annotation (Stage 2) ---

ANNOTATE_DIR = DATA / "annotate"

# Choice: 1 FPS for annotation windows.
# Alternative was 2 FPS — doubles tokens per window for marginal temporal
# resolution gain. Factory actions span seconds, 1 FPS captures transitions.
ANNOTATE_FPS = 1

# Choice: 5-frame windows (5 seconds at 1 FPS).
# Alternative was 3 frames — too few to see action start/end.
# 10 frames would exceed Qwen2.5-VL's practical image throughput per call.
WINDOW_SIZE = 5

# Choice: Qwen2.5-VL-7B-Instruct in fp16 (~16 GB VRAM).
ANNOTATE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Choice: max_new_tokens=512 for structured JSON response.
# Alternative was 256 — too tight for ECoT reasoning chain.
# 1024 wastes time on padding. 512 fits scene graph + reasoning + bbox.
MAX_NEW_TOKENS = 512
