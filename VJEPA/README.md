V-JEPA 2 encoding and diagnostics. Run from EgoClean root after Phase 1.

`encode.py` — GPU. Slides 64-frame window at 1-sec stride, saves per-clip embeddings.
`analyze.py` — CPU. Cosine distance, self-similarity matrix, autocorrelation. Outputs PNGs.
