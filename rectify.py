import json
import cv2
import numpy as np
import config

map_cache = {}

def load_intrinsics():
    with open(config.INTRINSICS_PATH) as f:
        d = json.load(f)
    K = np.array([
        [d["fx"], 0, d["cx"]],
        [0, d["fy"], d["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    D = np.array([[d["k1"], d["k2"], d["k3"], d["k4"]]], dtype=np.float64)
    size = (d["image_width"], d["image_height"])
    return K, D, size

def scale_K(K, from_size, to_size):
    w_from, h_from = from_size
    w_to, h_to = to_size
    s = np.array([[w_to / w_from, 0, 0], [0, h_to / h_from, 0], [0, 0, 1]], dtype=np.float64)
    return s @ K

def rectify_frame(frame, K, D, intrinsics_size):
    h, w = frame.shape[:2]
    frame_size = (w, h)
    if frame_size != intrinsics_size:
        K = scale_K(K, intrinsics_size, frame_size)
    key = (w, h)
    if key not in map_cache:
        R = np.eye(3, dtype=np.float64)
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), R, balance=config.FISHEYE_BALANCE)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, R, newK, (w, h), cv2.CV_16SC2)
        map_cache[key] = (map1, map2)
    map1, map2 = map_cache[key]
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

def iter_frames(video_path, rectify=True):
    out = load_intrinsics() if rectify else (None, None, None)
    K, D, size = out if rectify else (None, None, None)
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if rectify:
            frame = rectify_frame(frame, K, D, size)
        yield frame
    cap.release()
