import json
import os
import subprocess
import sys

import cv2
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from config import (
    RAW,
    HF_REPO,
    HF_FILES,
    TAR_DIR,
    POC_CLIPS,
    INTRINSICS_PATH,
    DEBUG_DIR,
)


def download():
    load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY")
    if not token:
        raise ValueError("HF_TOKEN not set. Use: export HF_TOKEN=your_token")
    RAW.mkdir(parents=True, exist_ok=True)
    for f in HF_FILES:
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f,
            local_dir=RAW,
            repo_type="dataset",
            token=token,
        )
    print(f"downloaded to {RAW}")


def extract():
    tar_path = TAR_DIR / "part000.tar"
    members = [f"{base}.mp4" for base in POC_CLIPS] + [f"{base}.json" for base in POC_CLIPS]
    subprocess.run(["tar", "-xf", str(tar_path)] + members, cwd=TAR_DIR, check=True)
    tar_path.unlink()
    print(f"extracted to {TAR_DIR}")


def load_intrinsics(path):
    raw = json.loads(path.read_text())
    K = np.array([
        [raw["fx"], 0.0, raw["cx"]],
        [0.0, raw["fy"], raw["cy"]],
        [0.0, 0.0, 1.0],
    ])
    D = np.array([raw["k1"], raw["k2"], raw["k3"], raw["k4"]])
    calib_size = (raw["image_width"], raw["image_height"])
    return K, D, calib_size


def scale_intrinsics(K, calib_size, frame_size):
    sx = frame_size[0] / calib_size[0]
    sy = frame_size[1] / calib_size[1]
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    return K_scaled


def iter_frames(video_path, K, D, calib_size, fps=None):
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (w, h)

    K_use = scale_intrinsics(K, calib_size, frame_size) if frame_size != calib_size else K

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_use, D, np.eye(3), K_use, frame_size, cv2.CV_16SC2,
    )

    step_ms = (1000.0 / fps) if fps else (1000.0 / src_fps)
    next_ms = 0.0
    idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, next_ms)
        ret, frame = cap.read()
        if not ret:
            break
        rect = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        yield idx, next_ms, rect
        idx += 1
        next_ms += step_ms

    cap.release()


def check_rectify():
    K, D, calib_size = load_intrinsics(INTRINSICS_PATH)
    clip = POC_CLIPS[0]
    video_path = TAR_DIR / f"{clip}.mp4"

    cap = cv2.VideoCapture(str(video_path))
    ret, raw_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"cannot read {video_path}")

    gen = iter_frames(video_path, K, D, calib_size, fps=1)
    _, _, rect_frame = next(gen)

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DEBUG_DIR / "raw.jpg"), raw_frame)
    cv2.imwrite(str(DEBUG_DIR / "rectified.jpg"), rect_frame)
    print(f"saved raw.jpg and rectified.jpg to {DEBUG_DIR}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"
    if cmd == "download":
        download()
    elif cmd == "extract":
        extract()
    elif cmd == "check-rectify":
        check_rectify()
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)
