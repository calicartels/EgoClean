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

# Choice: 1 FPS — feeds ~180 frames per clip (~13k visual tokens).
# Alternative was 2 FPS (~26k tokens) — still fits 32k context but leaves less
# headroom for the response and doubles ViT compute for marginal temporal gain.
ANNOTATE_FPS = 1

# Choice: Qwen2.5-VL-7B-Instruct (~16 GB VRAM in fp16).
# Alternative was 3B (~10 GB) — faster but weaker at multi-object reasoning
# and temporal localization. 7B leaves ~8 GB headroom on 24 GB card.
# At 456x256 frames the visual encoder overhead is small, so 7B fits.
ANNOTATE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Choice: max_new_tokens=4096 for full-clip action list with multi-object schema.
# Alternative was 2048 — too tight when clips produce 15+ detailed segments with
# ECoT fields, objects arrays, and bboxes; output truncates mid-JSON.
MAX_NEW_TOKENS = 4096