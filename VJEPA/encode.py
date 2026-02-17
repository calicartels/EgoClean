import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
from tqdm import tqdm

import config


def load_model():
    # Choice: fp16 + SDPA. Halves VRAM (~1.5 GB) and 2x faster on 3090.
    # Alternative: fp32 (~3 GB). Use fp32 only if fp16 throws NaN.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoVideoProcessor.from_pretrained(config.VJEPA_REPO)
    model = AutoModel.from_pretrained(
        config.VJEPA_REPO,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval()
    return processor, model, device


def encode_clip(video_path, processor, model, device):
    vr = VideoDecoder(str(video_path))
    fps = vr.metadata.average_fps or 30
    n_frames = vr.metadata.num_frames
    stride = max(1, int(fps * config.VJEPA_STRIDE_SEC))
    n_windows = max(0, (n_frames - config.VJEPA_WINDOW) // stride + 1)
    if n_windows == 0:
        print(f"clip too short: {n_frames} frames < {config.VJEPA_WINDOW} window")
        sys.exit(1)

    print(f"  {n_frames} frames, {fps:.1f} fps, {n_windows} windows")
    dtype = torch.float16 if device != "cpu" else torch.float32
    embs, tembs, timestamps = [], [], []

    for i in tqdm(range(n_windows), desc="encode", unit="win"):
        start = i * stride
        indices = np.arange(start, start + config.VJEPA_WINDOW)
        clip = vr.get_frames_at(indices=indices).data

        inputs = processor(clip, return_tensors="pt")
        for k in inputs:
            if inputs[k].is_floating_point():
                inputs[k] = inputs[k].to(device=device, dtype=dtype)
            else:
                inputs[k] = inputs[k].to(device)

        with torch.no_grad():
            features = model.get_vision_features(**inputs)

        # features shape: (1, T_patches * S_patches, 1024) = (1, 8192, 1024)
        # T_patches=32 temporal, S_patches=256 spatial for ViT-L fpc64-256
        f = features[0].cpu().float()

        # Choice: mean pool everything → 1024-dim. Works for anomaly detection.
        emb = f.mean(dim=0).numpy()

        # Choice: reshape to (32, 256, 1024), mean over 256 spatial → (32, 1024).
        # Preserves sub-window temporal structure at ~66ms resolution.
        # Alternative: keep full (32, 256, 1024) for spatial analysis too.
        # That's 32×256×1024×4 bytes × 1200 windows = ~40 GB. Not worth it yet.
        temb = f.reshape(config.VJEPA_T_PATCHES, config.VJEPA_S_PATCHES, -1).mean(dim=1).numpy()

        embs.append(emb)
        tembs.append(temb)
        timestamps.append(start / fps)

    return np.array(embs), np.array(tembs), np.array(timestamps)


clips = sorted(config.OUT.glob("rectified_clip_*.mp4"))
if not clips:
    print(f"no rectified clips in {config.OUT}")
    sys.exit(1)

to_encode = [p for p in clips if not (config.OUT / f"{p.stem}_emb.npy").exists()]
if not to_encode:
    print("all clips already encoded")
    sys.exit(0)

print(f"loading {config.VJEPA_REPO}")
processor, model, device = load_model()
print(f"  device: {device}")

for clip_path in tqdm(to_encode, desc="clips", unit="clip"):
    print(f"\n{clip_path.name}")
    embs, tembs, ts = encode_clip(clip_path, processor, model, device)

    stem = clip_path.stem
    np.save(config.OUT / f"{stem}_emb.npy", embs)
    np.save(config.OUT / f"{stem}_temb.npy", tembs)
    np.save(config.OUT / f"{stem}_ts.npy", ts)
    print(f"  saved {embs.shape} mean-pooled + {tembs.shape} temporal")

print("done")