V-JEPA 2 encoding, anomaly detection, and diagnostics. Run from EgoClean root after Phase 1.

`encode.py` — GPU. Slides 64-frame window at 1-sec stride, saves mean-pooled and temporal-token embeddings.
`detect.py` — CPU. Thresholds temporal-token similarity to find anomaly boundaries. Outputs anomalies.json.
`analyze.py` — CPU. Plots similarity signal with anomaly overlays, similarity matrix, and work/anomaly timeline strip.
