import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from VJEPA.utils import load_clip, sim_matrix, sim_matrix_mean, row_mean_score, fmt


def ema_smooth(signal, alpha):
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1]
    return out


def otsu_1d(signal):
    lo, hi = float(signal.min()), float(signal.max())
    if hi - lo < 1e-10:
        return (lo + hi) / 2
    normed = ((signal - lo) / (hi - lo) * 255).astype(np.uint8)
    thresh_int, _ = cv2.threshold(normed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return lo + (thresh_int / 255.0) * (hi - lo)


def hysteresis_detect(signal, theta_on, theta_off, debounce_on, debounce_off):
    n = len(signal)
    on = False
    run = 0
    states = np.zeros(n, dtype=bool)

    for i in range(n):
        if not on:
            if signal[i] > theta_on:
                run += 1
                if run >= debounce_on:
                    on = True
                    states[i - debounce_on + 1 : i + 1] = True
                    run = 0
            else:
                run = 0
        else:
            states[i] = True
            if signal[i] < theta_off:
                run += 1
                if run >= debounce_off:
                    on = False
                    states[i - debounce_off + 1 : i + 1] = False
                    run = 0
            else:
                run = 0
    return states


def states_to_segments(states, ts, min_dur):
    segs = []
    start = None
    for i, s in enumerate(states):
        if s and start is None:
            start = i
        elif not s and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(states)))

    result = []
    for s, e in segs:
        ec = min(e, len(ts) - 1)
        dur = ts[ec] - ts[s]
        if dur >= min_dur:
            result.append({
                "start_sec": round(float(ts[s]), 1),
                "end_sec": round(float(ts[ec]), 1),
                "duration": round(float(dur), 1),
                "start_fmt": fmt(ts[s]),
                "end_fmt": fmt(ts[ec]),
            })
    return result


def plot_detection(ts, raw, smoothed, theta_on, theta_off, median, anomalies, name):
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(ts, raw, alpha=0.25, linewidth=0.4, color="gray", label="raw")
    ax.plot(ts, smoothed, linewidth=0.8, color="#2b6cb0", label="EMA smoothed")
    ax.axhline(theta_on, color="#e53e3e", linewidth=0.8, linestyle="--",
               label=f"θ_on={theta_on:.4f} (Otsu)")
    ax.axhline(theta_off, color="#ed8936", linewidth=0.8, linestyle=":",
               label=f"θ_off={theta_off:.4f}")
    ax.axhline(median, color="#38a169", linewidth=0.6, linestyle="-.",
               label=f"median={median:.4f}", alpha=0.6)
    for a in anomalies:
        ax.axvspan(a["start_sec"], a["end_sec"], alpha=0.15, color="#e53e3e")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("anomaly score (1 − typicality)")
    ax.set_title(f"{name} — hysteresis detection")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(ts[0], ts[-1])
    plt.tight_layout()
    out = config.OUT / f"{name}_detection.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  plot: {out}")


# --- main ---

clip_indices = [i for i in range(1, config.EXPECTED_CLIPS + 1)
                if (config.OUT / f"rectified_clip_{i}_emb.npy").exists()]
if not clip_indices:
    print("no embeddings found, run encode.py first")
    sys.exit(1)

results = {}
for i in clip_indices:
    emb, temb, ts = load_clip(i)
    name = f"rectified_clip_{i}"

    S = sim_matrix(temb) if temb is not None else sim_matrix_mean(emb)
    typicality = row_mean_score(S)
    raw_score = 1.0 - typicality

    smoothed = ema_smooth(raw_score, config.DETECT_EMA_ALPHA)

    theta_on = config.DETECT_THETA_ON if config.DETECT_THETA_ON is not None else otsu_1d(smoothed)
    median = float(np.median(smoothed))
    theta_off = median + config.DETECT_HYSTERESIS_RATIO * (theta_on - median)

    states = hysteresis_detect(
        smoothed, theta_on, theta_off,
        config.DETECT_DEBOUNCE_ON, config.DETECT_DEBOUNCE_OFF,
    )
    anomalies = states_to_segments(states, ts, config.DETECT_MIN_DUR)

    results[name] = {
        "theta_on": round(float(theta_on), 4),
        "theta_off": round(float(theta_off), 4),
        "median": round(float(median), 4),
        "method": "otsu" if config.DETECT_THETA_ON is None else "manual",
        "anomalies": anomalies,
    }

    print(f"{name}: median={median:.4f}, θ_on={theta_on:.4f}, θ_off={theta_off:.4f}, {len(anomalies)} anomalies")
    for a in anomalies:
        print(f"  {a['start_fmt']}-{a['end_fmt']} ({a['duration']}s)")

    plot_detection(ts, raw_score, smoothed, theta_on, theta_off, median, anomalies, name)

out_path = config.OUT / "anomalies.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {out_path}")
