import json
import cv2
from config import TAR_DIR, POC_CLIPS, INTRINSICS_PATH, DEBUG_DIR, TRIAGE_DIR
from ingest import load_intrinsics, iter_frames

K, D, calib_size = load_intrinsics(INTRINSICS_PATH)
clip = POC_CLIPS[0]
video_path = TAR_DIR / f"{clip}.mp4"
detections = json.loads((TRIAGE_DIR / f"{clip}_detections.json").read_text())

# Sample 5 frames spread across the clip
sample_idxs = {0, 30, 60, 90, 120}
frames = {}
for idx, ts_ms, frame in iter_frames(video_path, K, D, calib_size, fps=1):
    if idx in sample_idxs:
        frames[idx] = frame
    if idx > max(sample_idxs):
        break

DEBUG_DIR.mkdir(parents=True, exist_ok=True)

for idx in sorted(frames):
    frame = frames[idx].copy()
    det = detections[idx]
    for box, label, score in zip(det["boxes"], det["labels"], det["scores"]):
        x1, y1, x2, y2 = [int(v) for v in box]
        # green for hands, red for objects
        if "hand" in label.lower():
            color = (0, 255, 0)
            tag = f"H {score:.2f}"
        else:
            color = (0, 0, 255)
            tag = f"O {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, tag, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    cv2.imwrite(str(DEBUG_DIR / f"triage_frame_{idx:03d}.jpg"), frame)

print(f"saved {len(frames)} annotated frames to {DEBUG_DIR}")