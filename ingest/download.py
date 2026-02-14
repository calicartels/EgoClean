import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from config import RAW, HF_REPO, HF_FILES

load_dotenv()
token = os.getenv("HF_KEY")
if not token:
    raise ValueError("HF_KEY not set in .env")

RAW.mkdir(parents=True, exist_ok=True)

# part000.tar is ~1 GB; intrinsics.json is tiny.
for f in HF_FILES:
    hf_hub_download(
        repo_id=HF_REPO,
        filename=f,
        local_dir=RAW,
        repo_type="dataset",
        token=token,
    )
print(RAW)
