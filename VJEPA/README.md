V-JEPA 2 encoding, anomaly detection, and diagnostics. Run from EgoClean root after Phase 1.

`encode.py` — GPU. Slides 64-frame window at 1-sec stride, saves mean-pooled and temporal-token embeddings.
`detect.py` — CPU. Hysteresis anomaly detector (LAPS-style EMA + Otsu + dual-threshold state machine). Outputs anomalies.json + per-clip detection plot.
`analyze.py` — CPU. 4-panel diagnostics: consecutive distance, similarity matrix, autocorrelation, temporal-token similarity.
`utils.py` — Shared loading and similarity functions.
