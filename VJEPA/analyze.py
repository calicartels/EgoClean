import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from VJEPA.utils import load_clip, sim_matrix, sim_matrix_mean, row_mean_score, fmt

# Choice: inferno — perceptually uniform, high contrast at boundaries.
# Alternative: viridis (low contrast in 0.7-0.9 range), magma.
CMAP = "inferno"


def plot_clip(emb, temb, ts, name):
    has_temb = temb is not None
    S = sim_matrix(temb) if has_temb else sim_matrix_mean(emb)
    label = "temporal-token" if has_temb else "mean-pooled"
    score = row_mean_score(S)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [1, 1.5]})

    ax = axes[0]
    ax.fill_between(ts, score, alpha=0.15, color="#4a5568")
    ax.plot(ts, score, linewidth=0.7, color="#2d3748")

    p10 = np.percentile(score, 10)
    low = score < p10
    ax.fill_between(ts, score, where=low, alpha=0.35, color="#c53030")
    ax.axhline(y=p10, color="#c53030", linewidth=0.7, linestyle=":", alpha=0.5)

    in_dip = False
    dip_start = 0
    for i, is_low in enumerate(low):
        if is_low and not in_dip:
            dip_start = i
            in_dip = True
        elif not is_low and in_dip:
            mid = (ts[dip_start] + ts[i - 1]) / 2
            y = score[dip_start:i].min()
            ax.annotate(f"{fmt(ts[dip_start])}-{fmt(ts[i-1])}",
                        xy=(mid, y), fontsize=8, ha="center", va="top",
                        color="#c53030", weight="bold")
            in_dip = False
    if in_dip:
        mid = (ts[dip_start] + ts[-1]) / 2
        ax.annotate(f"{fmt(ts[dip_start])}-{fmt(ts[-1])}",
                    xy=(mid, score[dip_start:].min()),
                    fontsize=8, ha="center", va="top",
                    color="#c53030", weight="bold")

    ax.set_ylabel("row-mean similarity")
    ax.set_xlabel("time (s)")
    ax.set_title(f"{name} — typicality score (low = anomalous)")
    ax.set_xlim(ts[0], ts[-1])

    ax = axes[1]
    im = ax.imshow(S, aspect="auto", cmap=CMAP,
                   extent=[ts[0], ts[-1], ts[-1], ts[0]])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"{name} — {label} similarity matrix")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    plt.tight_layout()
    out = config.OUT / f"{name}_diagnostics.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


clip_indices = [i for i in range(1, config.EXPECTED_CLIPS + 1)
                if (config.OUT / f"rectified_clip_{i}_emb.npy").exists()]
if not clip_indices:
    print(f"no embeddings in {config.OUT}")
    sys.exit(1)

for i in tqdm(clip_indices, desc="analyze", unit="clip"):
    emb, temb, ts = load_clip(i)
    name = f"rectified_clip_{i}"
    print(f"{name}: {emb.shape[0]} windows")
    plot_clip(emb, temb, ts, name)
