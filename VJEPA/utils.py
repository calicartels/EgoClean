import numpy as np
import config


def load_clip(clip_idx):
    stem = f"rectified_clip_{clip_idx}"
    emb = np.load(config.OUT / f"{stem}_emb.npy")
    ts = np.load(config.OUT / f"{stem}_ts.npy")
    temb_path = config.OUT / f"{stem}_temb.npy"
    temb = np.load(temb_path) if temb_path.exists() else None
    return emb, temb, ts


def sim_matrix(temb):
    n = temb.shape[0]
    flat = temb.reshape(n, -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normed = flat / norms
    return normed @ normed.T


def sim_matrix_mean(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / norms
    return normed @ normed.T


def row_mean_score(S):
    n = S.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.array([S[i, mask[i]].mean() for i in range(n)])


def fmt(sec):
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"
