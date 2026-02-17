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
    embeddings = []
    timestamps = []

    for i in tqdm(range(n_windows), desc="encode", unit="win"):
        start = i * stride
        indices = np.arange(start, start + config.VJEPA_WINDOW)

        # Choice: decode 64 frames per call. ~20 MB per window at 1080p.
        # Alternative: preload all frames (~33 GB). Killed the process.
        clip = vr.get_frames_at(indices=indices).data  # (64, C, H, W) uint8

        inputs = processor(clip, return_tensors="pt")
        for k in inputs:
            if inputs[k].is_floating_point():
                inputs[k] = inputs[k].to(device=device, dtype=dtype)
            else:
                inputs[k] = inputs[k].to(device)

        with torch.no_grad():
            features = model.get_vision_features(**inputs)

        # Choice: mean pool over all patch tokens → single 1024-dim vector.
        # Alternative: keep full (8192, 1024) per window. ~6 GB per clip.
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

    embeddings, timestamps = encode_clip(clip_path, processor, model, device)

    stem = clip_path.stem
    np.save(config.OUT / f"{stem}_emb.npy", embeddings)
    np.save(config.OUT / f"{stem}_ts.npy", timestamps)
    print(f"  saved {embeddings.shape[0]} embeddings {embeddings.shape}")

print("\ndone")
