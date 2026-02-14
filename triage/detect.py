import json

import cv2
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from config import (
    TAR_DIR,
    POC_CLIPS,
    INTRINSICS_PATH,
    TRIAGE_DIR,
    TRIAGE_FPS,
    TRIAGE_PROMPT,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
)
from ingest import load_intrinsics, iter_frames

MODEL_ID = "IDEA-Research/grounding-dino-base"


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    return processor, model


def detect_frame(frame, processor, model):
    device = next(model.parameters()).device
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, text=TRIAGE_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(frame.shape[0], frame.shape[1])],
    )[0]
    boxes = results["boxes"].cpu().numpy().tolist()
    scores = results["scores"].cpu().numpy().tolist()
    labels = [str(l) for l in results["labels"]]
    return boxes, scores, labels


def run():
    processor, model = load_model()
    K, D, calib_size = load_intrinsics(INTRINSICS_PATH)
    TRIAGE_DIR.mkdir(parents=True, exist_ok=True)

    for clip in POC_CLIPS:
        video_path = TAR_DIR / f"{clip}.mp4"
        if not video_path.exists():
            print(f"skip {clip}: not found", file=__import__("sys").stderr)
            continue
        detections = []
        for idx, ts_ms, frame in iter_frames(video_path, K, D, calib_size, fps=TRIAGE_FPS):
            boxes, scores, labels = detect_frame(frame, processor, model)
            detections.append({
                "frame_idx": idx,
                "timestamp_ms": ts_ms,
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })
        out = TRIAGE_DIR / f"{clip}_detections.json"
        out.write_text(json.dumps(detections))

    del model, processor
    torch.cuda.empty_cache()
    n = sum(
        len(json.loads((TRIAGE_DIR / f"{c}_detections.json").read_text()))
        for c in POC_CLIPS
        if (TRIAGE_DIR / f"{c}_detections.json").exists()
    )
    print(f"{n} frames detected across {len(POC_CLIPS)} clips -> {TRIAGE_DIR}")


if __name__ == "__main__":
    run()
