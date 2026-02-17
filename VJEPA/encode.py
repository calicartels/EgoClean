import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
from tqdm import tqdm

import config


def read_frames(video_path):
    vr = VideoDecoder(str(video_path))
    fps = vr.metadata.average_fps or 30
    n_frames = vr.metadata.num_frames
    indices = np.arange(n_frames)
    frames = vr.get_frames_at(indices=indices).data  # (N, C, H, W) uint8
    if len(frames) == 0:
        print(f"no frames read from {video_path}")
        sys.exit(1)
    # Resize to avoid OOM: 1080p × 5400 frames = ~33 GB. 256² = ~1 GB.
    frames = frames.float()
    frames = F.interpolate(frames, size=(config.VJEPA_RESIZE, config.VJEPA_RESIZE), mode="bilinear")
    frames = frames.clamp(0, 255).byte()
    return frames, fps


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


def encode_clip(frames, fps, processor, model, device):
    stride = max(1, int(fps * config.VJEPA_STRIDE_SEC))
    n_frames = len(frames)
    n_windows = max(0, (n_frames - config.VJEPA_WINDOW) // stride + 1)
    if n_windows == 0:
        print(f"clip too short: {n_frames} frames < {config.VJEPA_WINDOW} window")
        sys.exit(1)

    embeddings = []
    timestamps = []

    # Choice: batch_size=1 per window. Simple, ~0.3 sec/window on 3090.
    # Alternative: batch=4-8 would be ~2x faster but adds padding logic.
    for i in tqdm(range(n_windows), desc="encode", unit="win"):
        start = i * stride
        clip = frames[start : start + config.VJEPA_WINDOW]  # (64, C, H, W) uint8

        inputs = processor(clip, return_tensors="pt")
        for k in inputs:
            if inputs[k].is_floating_point():
                inputs[k] = inputs[k].to(device=device, dtype=torch.float16 if device != "cpu" else torch.float32)
            else:
                inputs[k] = inputs[k].to(device)

        with torch.no_grad():
            features = model.get_vision_features(**inputs)

        # Choice: mean pool over all 8192 patch tokens → single 1024-dim vector.
        # Alternative: keep full (8192, 1024) per window for spatial analysis.
        # That's ~6 GB per clip — not worth storing for diagnostics.
        emb = features.mean(dim=1).cpu().float().numpy()[0]
        embeddings.append(emb)
        timestamps.append(start / fps)

    return np.array(embeddings), np.array(timestamps)


clips = sorted(config.OUT.glob("rectified_clip_*.mp4"))
if not clips:
    print(f"no rectified clips in {config.OUT}")
    sys.exit(1)

print(f"loading {config.VJEPA_REPO}")
processor, model, device = load_model()
print(f"  device: {device}")

for clip_path in tqdm(clips, desc="clips", unit="clip"):
    print(f"\n{clip_path.name}")
    frames, fps = read_frames(clip_path)
    print(f"  {len(frames)} frames, {fps:.1f} fps")

    embeddings, timestamps = encode_clip(frames, fps, processor, model, device)

    stem = clip_path.stem
    np.save(config.OUT / f"{stem}_emb.npy", embeddings)
    np.save(config.OUT / f"{stem}_ts.npy", timestamps)
    print(f"  saved {embeddings.shape[0]} embeddings {embeddings.shape}")

    del frames

print("\ndone")
