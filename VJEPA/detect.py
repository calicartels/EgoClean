import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm
import config
from VJEPA.utils import load_clip, sim_matrix, sim_matrix_mean, row_mean_score, fmt


def find_anomalies(score, ts, percentile, min_gap, min_dur):
    # Choice: percentile-based threshold. Nonparametric, adapts per clip.
    # Alternative: mean-k*std (failed on consecutive sim), fixed threshold.
    threshold = np.percentile(score, percentile)
    low = score < threshold

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
                "mean_score": round(float(score[s:e].mean()), 4),
            })
    return result, threshold


clip_indices = [i for i in range(1, config.EXPECTED_CLIPS + 1)
                if (config.OUT / f"rectified_clip_{i}_emb.npy").exists()]
if not clip_indices:
    print("no embeddings found, run encode.py first")
    sys.exit(1)
results = {}
for i in tqdm(clip_indices, desc="detect", unit="clip"):
    emb, temb, ts = load_clip(i)
    name = f"rectified_clip_{i}"

    S = sim_matrix(temb) if temb is not None else sim_matrix_mean(emb)
    score = row_mean_score(S)

    anomalies, threshold = find_anomalies(
        score, ts,
        percentile=config.ANOMALY_PERCENTILE,
        min_gap=config.ANOMALY_MIN_GAP,
        min_dur=config.ANOMALY_MIN_DUR,
    )

    results[name] = {
        "threshold": round(float(threshold), 4),
        "score_mean": round(float(score.mean()), 4),
        "score_std": round(float(score.std()), 4),
        "anomalies": anomalies,
    }

    print(f"{name}: threshold={threshold:.4f}, {len(anomalies)} anomalies")
    for a in anomalies:
        print(f"  {fmt(a['start_sec'])}-{fmt(a['end_sec'])}  "
              f"({a['start_sec']}-{a['end_sec']}s, score={a['mean_score']:.4f})")

out_path = config.OUT / "anomalies.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved {out_path}")
