import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from VJEPA.utils import load_clip, sim_matrix, sim_matrix_mean, row_mean_score

# Choice: inferno — high contrast at the 0.6-0.9 similarity range where our data lives.
# Alternative: viridis (default, washes out in that range), plasma (similar to inferno
# but worse at extremes), magma (too dark for the matrix background).
CMAP = "inferno"


def cosine_dist(emb):
    a, b = emb[:-1], emb[1:]
    dots = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return 1.0 - dots / norms


def autocorr(x, max_lag=None):
    x = x - x.mean()
    if max_lag is None:
        max_lag = len(x) // 2
    result = np.correlate(x, x, mode="full")
    result = result[len(result) // 2:]
    if result[0] != 0:
        result = result / result[0]
    return result[:max_lag]


def plot_clip(emb, temb, ts, name):
    has_temb = temb is not None
    S = sim_matrix(temb) if has_temb else sim_matrix_mean(emb)
    source = "temporal-token" if has_temb else "mean-pooled"

    dist = cosine_dist(emb)
    score = row_mean_score(S)
    ac = autocorr(score, max_lag=min(300, len(score) // 2))

    fig, axes = plt.subplots(4, 1, figsize=(16, 16),
                             gridspec_kw={"height_ratios": [1, 1, 1.5, 1]})

    # --- Panel 1: Consecutive cosine distance ---
    ax = axes[0]
    ax.plot(ts[1:], dist, linewidth=0.5, color="#2b6cb0")
    ax.set_ylabel("cosine distance")
    ax.set_title(f"{name} — consecutive distance (spikes = scene change)")
    ax.set_xlim(ts[0], ts[-1])

    # --- Panel 2: Row-mean typicality score ---
    ax = axes[1]
    ax.plot(ts, score, linewidth=0.6, color="#2d3748")
    ax.fill_between(ts, score, alpha=0.1, color="#4a5568")
    ax.set_ylabel("row-mean similarity")
    ax.set_title(f"{name} — typicality (low = unlike the rest of the clip)")
    ax.set_xlim(ts[0], ts[-1])

    # --- Panel 3: Similarity matrix ---
    ax = axes[2]
    im = ax.imshow(S, aspect="auto", cmap=CMAP,
                   extent=[ts[0], ts[-1], ts[-1], ts[0]])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"{name} — {source} similarity matrix")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # --- Panel 4: Autocorrelation of typicality score ---
    ax = axes[3]
    lags = np.arange(len(ac))
    ax.plot(lags, ac, linewidth=0.6, color="#38a169")
    ax.set_xlabel("lag (windows)")
    ax.set_ylabel("autocorrelation")
    ax.set_title(f"{name} — autocorrelation of typicality score")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlim(0, len(ac))

    plt.tight_layout()
    out = config.OUT / f"{name}_diagnostics.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


found = False
for i in tqdm(range(1, config.EXPECTED_CLIPS + 1), desc="analyze"):
    path = config.OUT / f"rectified_clip_{i}_emb.npy"
    if not path.exists():
        continue
    found = True
    emb, temb, ts = load_clip(i)
    name = f"rectified_clip_{i}"
    print(f"{name}: {emb.shape[0]} windows")
    plot_clip(emb, temb, ts, name)

if not found:
    print(f"no embeddings in {config.OUT}")
    sys.exit(1)