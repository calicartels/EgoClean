from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLIP_DIR = DATA / "factory_001"
OUT_DIR = DATA / "qwen_labels"

CLIPS = sorted(CLIP_DIR.glob("rectified_clip_*.mp4"))

# Choice: Qwen2.5-VL-7B-Instruct (~16 GB fp16).
# Alternative: 3B (~10 GB) — faster but weaker at temporal reasoning.
# 7B leaves ~8 GB headroom on 24 GB card.
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Choice: 30-second windows at 1fps = 30 frames per VLM call.
# At ~250 visual tokens/frame = ~7500 visual tokens, fits 32k context.
# Alternative: 60s (60 frames, ~15k tokens) — tighter fit, model attention
# degrades for fine-grained per-second labels at longer sequences.
WINDOW_SEC = 30
STRIDE_SEC = 30
SAMPLE_FPS = 1

# Choice: 512 new tokens. Transition-only output means 1-5 lines per window.
# Even worst case (per-second labels) is ~150 tokens for 30 lines.
# Alternative: 256 (risks truncation if model adds preamble).
MAX_TOKENS = 512