import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def load_clip(clip_idx):
    stem = f"rectified_clip_{clip_idx}"
    emb = np.load(config.OUT / f"{stem}_emb.npy")
    ts = np.load(config.OUT / f"{stem}_ts.npy")
    return emb, ts


def cosine_dist(emb):
    a = emb[:-1]
    b = emb[1:]
    dots = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return 1.0 - dots / norms


def sim_matrix(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / norms
    return normed @ normed.T


def autocorr(signal):
    x = signal - signal.mean()
    result = np.correlate(x, x, mode="full")
    result = result[len(x) - 1 :]
    if result[0] != 0:
        result /= result[0]
    return result


def plot_diagnostics(emb, ts, name):
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    stride_sec = config.VJEPA_STRIDE_SEC

    dist = cosine_dist(emb)
    axes[0].plot(ts[1:], dist, linewidth=0.6, color="#2b6cb0")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("cosine distance")
    axes[0].set_title(f"{name} — consecutive cosine distance (high = change)")

    sim = sim_matrix(emb)
    im = axes[1].imshow(
        sim, aspect="auto", cmap="viridis",
        extent=[ts[0], ts[-1], ts[-1], ts[0]],
    )
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("time (s)")
    axes[1].set_title(f"{name} — self-similarity (bright = similar content)")
    plt.colorbar(im, ax=axes[1])

    ac = autocorr(dist)
    lags = np.arange(len(ac)) * stride_sec
    half = len(ac) // 2
    axes[2].plot(lags[:half], ac[:half], linewidth=0.6, color="#9b2c2c")
    axes[2].set_xlabel("lag (s)")
    axes[2].set_ylabel("autocorrelation")
    axes[2].set_title(f"{name} — autocorrelation (peaks = repeating period)")
    axes[2].axhline(y=0, color="gray", linewidth=0.5)

    plt.tight_layout()
    out_path = config.OUT / f"{name}_diagnostics.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path}")


found = False
for i in tqdm(range(1, config.EXPECTED_CLIPS + 1), desc="analyze", unit="clip"):
    path = config.OUT / f"rectified_clip_{i}_emb.npy"
    if not path.exists():
        continue
    found = True
    emb, ts = load_clip(i)
    name = f"rectified_clip_{i}"
    print(f"{name}: {emb.shape[0]} embeddings, {ts[0]:.1f}–{ts[-1]:.1f}s")
    plot_diagnostics(emb, ts, name)

if not found:
    print(f"no embeddings found in {config.OUT}")
    sys.exit(1)

print("done")
