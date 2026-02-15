import os
import re
import sys
import tarfile
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from tqdm import tqdm

import config

def download():
    load_dotenv()
    token = os.getenv("HF_KEY")
    config.RAW.mkdir(parents=True, exist_ok=True)
    for fn in config.HF_FILES:
        hf_hub_download(
            repo_id=config.HF_REPO,
            filename=fn,
            local_dir=config.RAW,
            repo_type="dataset",
            token=token,
        )
    print(config.RAW)

def rm_empty_dirs(p, stop_at):
    if p.exists() and p.is_dir() and not any(p.iterdir()) and p != stop_at:
        p.rmdir()
        if p.parent != stop_at:
            rm_empty_dirs(p.parent, stop_at)

def parse_video_idx(name):
    m = re.search(r"_(\d{5})\.mp4$", name)
    return int(m.group(1)) if m else 0

def extract():
    if not config.TAR_PATH.exists():
        sys.exit(1)
    config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(config.TAR_PATH) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".mp4")]
        members.sort(key=lambda m: parse_video_idx(m.name))
        if not members:
            return
        for i, m in enumerate(tqdm(members, desc="extract"), start=1):
            f = tar.extractfile(m)
            if f:
                (config.EXTRACT_DIR / f"data_point_{i:03d}.mp4").write_bytes(f.read())
    config.TAR_PATH.unlink()
    rm_empty_dirs(config.TAR_PATH.parent, config.EXTRACT_DIR)
    cache = config.RAW / ".cache"
    if cache.exists():
        shutil.rmtree(cache)
    print(config.EXTRACT_DIR)

cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
if cmd == "download":
    download()
elif cmd == "extract":
    extract()
elif cmd == "all":
    download()
    extract()
else:
    sys.exit(1)
