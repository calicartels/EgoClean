import os
import re
import subprocess
import sys
import tarfile
import shutil
import cv2
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from tqdm import tqdm

import config
from rectify import iter_frames

def is_done():
    if not (config.OUT / "intrinsics.json").exists():
        return False
    for i in range(1, config.EXPECTED_CLIPS + 1):
        if not (config.OUT / f"rectified_clip_{i}.mp4").exists():
            return False
    return True

def download():
    if config.TAR_PATH.exists():
        return
    print("download")
    load_dotenv()
    token = os.getenv("HF_KEY")
    config.DATA.mkdir(parents=True, exist_ok=True)
    for fn in tqdm(config.HF_FILES, desc="download"):
        hf_hub_download(
            repo_id=config.HF_REPO,
            filename=fn,
            local_dir=config.DATA,
            repo_type="dataset",
            token=token,
        )

def extract():
    mp4s = list(config.EXTRACT_DIR.glob("*.mp4"))
    if mp4s and config.INTRINSICS_PATH.exists():
        return
    print("extract")
    if not config.TAR_PATH.exists():
        sys.exit(1)
    config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(config.TAR_PATH) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".mp4")]
        members.sort(key=lambda m: int(re.search(r"_(\d{5})\.mp4$", m.name).group(1)) if re.search(r"_(\d{5})\.mp4$", m.name) else 99999)
        for i, m in enumerate(tqdm(members, desc="extract"), start=1):
            f = tar.extractfile(m)
            if f:
                (config.EXTRACT_DIR / f"data_point_{i:03d}.mp4").write_bytes(f.read())
    config.TAR_PATH.unlink()
    src = config.TAR_PATH.parent / "intrinsics.json"
    if src.exists():
        shutil.copy(src, config.INTRINSICS_PATH)
    workers_dir = config.TAR_PATH.parent.parent  # factory_001/workers
    if workers_dir.exists():
        shutil.rmtree(workers_dir)
    cache = config.DATA / ".cache"
    if cache.exists():
        shutil.rmtree(cache)

def rectify():
    if is_done():
        return
    print("rectify")
    if not config.INTRINSICS_PATH.exists():
        sys.exit(1)
    config.OUT.mkdir(parents=True, exist_ok=True)
    for i, mp4 in enumerate(tqdm(sorted(config.EXTRACT_DIR.glob("*.mp4")), desc="rectify"), start=1):
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        out = config.OUT / f"rectified_clip_{i}.mp4"
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-crf", str(config.VIDEO_CRF), "-preset", "medium", "-pix_fmt", "yuv420p",
                str(out),
            ],
            stdin=subprocess.PIPE,
        )
        for frame in iter_frames(mp4, rectify=True):
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
    shutil.copy(config.INTRINSICS_PATH, config.OUT / "intrinsics.json")
    shutil.rmtree(config.EXTRACT_DIR)
    flatten_out()

def flatten_out():
    for p in config.OUT.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        elif p.name.startswith("data_point_"):
            p.unlink()

cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
if cmd == "download":
    download()
elif cmd == "extract":
    extract()
elif cmd == "rectify":
    rectify()
elif cmd == "all":
    if is_done():
        flatten_out()
        print("already done")
        sys.exit(0)
    download()
    extract()
    rectify()
    print("done")
else:
    sys.exit(1)
