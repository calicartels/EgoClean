from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"


HF_REPO = "builddotai/Egocentric-100K"
HF_FILES = ("factory001/worker001/part000.tar", "factory001/worker001/intrinsics.json")
