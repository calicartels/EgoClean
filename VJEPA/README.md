V-JEPA 2 encoding, anomaly detection, and diagnostics. Run from EgoClean root after Phase 1.

`encode.py` — GPU. Slides 64-frame window at 1-sec stride, saves mean-pooled and temporal-token embeddings.
`detect.py` — CPU. Percentile-based anomaly detection on row-mean similarity. Outputs anomalies.json.
`analyze.py` — CPU. Plots typicality score and similarity matrix with anomaly shading.
