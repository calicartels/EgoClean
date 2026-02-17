import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import config


def load_clip(clip_idx):
    stem = f"rectified_clip_{clip_idx}"
    emb = np.load(config.OUT / f"{stem}_emb.npy")
    ts = np.load(config.OUT / f"{stem}_ts.npy")
    temb_path = config.OUT / f"{stem}_temb.npy"
    temb = np.load(temb_path) if temb_path.exists() else None
    return emb, temb, ts


def diagonal_similarity(temb):
    n = temb.shape[0]
    flat = temb.reshape(n, -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normed = flat / norms
    return np.sum(normed[:-1] * normed[1:], axis=1)


def find_anomalies(sim, ts, threshold, min_gap, min_dur):
    low = sim < threshold
    spans = []
    start = None
    for i, is_low in enumerate(low):
        if is_low and start is None:
            start = i
        elif not is_low and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(low)))

    merged = []
    for s, e in spans:
        if merged and (ts[s] - ts[merged[-1][1] - 1]) < min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    result = []
    for s, e in merged:
        dur = ts[min(e, len(ts) - 1)] - ts[s]
        if dur >= min_dur:
            result.append({
                "start_sec": round(float(ts[s]), 1),
                "end_sec": round(float(ts[min(e, len(ts) - 1)]), 1),
                "start_idx": int(s),
                "end_idx": int(e),
                "mean_sim": round(float(sim[s:e].mean()), 3),
            })
    return result


def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


results = {}
for i in range(1, config.EXPECTED_CLIPS + 1):
    path = config.OUT / f"rectified_clip_{i}_emb.npy"
    if not path.exists():
        continue
    emb, temb, ts = load_clip(i)
    name = f"rectified_clip_{i}"

    if temb is not None:
        sim = diagonal_similarity(temb)
    else:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        normed = emb / norms
        sim = np.sum(normed[:-1] * normed[1:], axis=1)

    # Choice: mean - 2*std. Adaptive across clips with different baseline similarity.
    # Alternative: fixed threshold (e.g. 0.85).
    threshold = float(sim.mean() - config.ANOMALY_N_STD * sim.std())

    anomalies = find_anomalies(
        sim, ts,
        threshold=threshold,
        min_gap=config.ANOMALY_MIN_GAP,
        min_dur=config.ANOMALY_MIN_DUR,
    )

    results[name] = {
        "threshold": round(threshold, 4),
        "sim_mean": round(float(sim.mean()), 4),
        "sim_std": round(float(sim.std()), 4),
        "anomalies": anomalies,
    }

    print(f"{name}: threshold={threshold:.4f}, {len(anomalies)} anomalies")
    for a in anomalies:
        print(f"  {fmt_time(a['start_sec'])}–{fmt_time(a['end_sec'])}  "
              f"({a['start_sec']}–{a['end_sec']}s, sim={a['mean_sim']:.3f})")

out_path = config.OUT / "anomalies.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {out_path}")
