import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def load_clip(clip_idx):
    stem = f"rectified_clip_{clip_idx}"
    emb = np.load(config.OUT / f"{stem}_emb.npy")
    ts = np.load(config.OUT / f"{stem}_ts.npy")
    temb_path = config.OUT / f"{stem}_temb.npy"
    temb = np.load(temb_path) if temb_path.exists() else None
    return emb, temb, ts


def sim_matrix_temporal(temb):
    n = temb.shape[0]
    flat = temb.reshape(n, -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normed = flat / norms
    return normed @ normed.T


def diag_sim(temb):
    n = temb.shape[0]
    flat = temb.reshape(n, -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normed = flat / norms
    return np.sum(normed[:-1] * normed[1:], axis=1)


def shade_anomalies(ax, anomalies):
    for a in anomalies:
        ax.axvspan(a["start_sec"], a["end_sec"], alpha=0.25, color="#e53e3e", zorder=0)


def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def plot_diagnostics(emb, temb, ts, name, anomalies):
    has_temb = temb is not None
    n_axes = 3 if has_temb else 2
    fig, axes = plt.subplots(n_axes, 1, figsize=(16, 5 * n_axes))

    if has_temb:
        sim = diag_sim(temb)
        threshold = float(sim.mean() - config.ANOMALY_N_STD * sim.std())
        axes[0].plot(ts[1:], sim, linewidth=0.5, color="#2b6cb0")
        axes[0].axhline(y=threshold, color="#e53e3e", linewidth=1, linestyle="--",
                       label=f"threshold={threshold:.3f}")
        shade_anomalies(axes[0], anomalies)
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("cosine similarity")
        axes[0].set_title(f"{name} — consecutive similarity (temporal tokens)")
        axes[0].legend(loc="lower left")
        for a in anomalies:
            mid = (a["start_sec"] + a["end_sec"]) / 2
            axes[0].annotate(
                f'{fmt_time(a["start_sec"])}–{fmt_time(a["end_sec"])}',
                xy=(mid, threshold), fontsize=8, ha="center", va="bottom",
                color="#e53e3e", weight="bold",
            )

    panel = 1 if has_temb else 0
    if has_temb:
        tsim = sim_matrix_temporal(temb)
        im = axes[panel].imshow(tsim, aspect="auto", cmap="viridis",
                                extent=[ts[0], ts[-1], ts[-1], ts[0]])
        axes[panel].set_title(f"{name} — temporal-token similarity matrix")
    else:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        sim_m = (emb / norms) @ (emb / norms).T
        im = axes[panel].imshow(sim_m, aspect="auto", cmap="viridis",
                               extent=[ts[0], ts[-1], ts[-1], ts[0]])
        axes[panel].set_title(f"{name} — mean-pooled similarity matrix")
    axes[panel].set_xlabel("time (s)")
    axes[panel].set_ylabel("time (s)")
    plt.colorbar(im, ax=axes[panel])
    for a in anomalies:
        for edge in [a["start_sec"], a["end_sec"]]:
            axes[panel].axhline(y=edge, color="#e53e3e", linewidth=0.8, alpha=0.7)
            axes[panel].axvline(x=edge, color="#e53e3e", linewidth=0.8, alpha=0.7)

    ax_strip = axes[2] if has_temb else axes[1]
    total = ts[-1]
    ax_strip.barh(0, total, height=0.6, color="#48bb78", label="work")
    for a in anomalies:
        dur = a["end_sec"] - a["start_sec"]
        ax_strip.barh(0, dur, left=a["start_sec"], height=0.6, color="#e53e3e", label="anomaly")
        ax_strip.text(a["start_sec"] + dur / 2, 0,
                      f'{fmt_time(a["start_sec"])}–{fmt_time(a["end_sec"])}',
                      ha="center", va="center", fontsize=9, color="white", weight="bold")
    ax_strip.set_xlim(0, total)
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("time (s)")
    ax_strip.set_title(f"{name} — work (green) vs anomaly (red)")
    handles, labels = ax_strip.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_strip.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    out_path = config.OUT / f"{name}_diagnostics.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path}")


anomaly_path = config.OUT / "anomalies.json"
if not anomaly_path.exists():
    print("run detect.py first")
    sys.exit(1)
with open(anomaly_path) as f:
    all_anomalies = json.load(f)

found = False
for i in range(1, config.EXPECTED_CLIPS + 1):
    path = config.OUT / f"rectified_clip_{i}_emb.npy"
    if not path.exists():
        continue
    found = True
    emb, temb, ts = load_clip(i)
    name = f"rectified_clip_{i}"
    anomalies = all_anomalies.get(name, {}).get("anomalies", [])
    print(f"{name}: {emb.shape[0]} emb, {len(anomalies)} anomalies")
    plot_diagnostics(emb, temb, ts, name, anomalies)

if not found:
    print(f"no embeddings found in {config.OUT}")
    sys.exit(1)
print("done")
