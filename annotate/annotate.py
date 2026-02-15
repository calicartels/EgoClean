import json
import cv2
import torch
from PIL import Image

# Choice: Qwen2_5_VLForConditionalGeneration + process_vision_info from qwen_vl_utils.
# Alternative was using pipeline() API — less control over multi-image batching
# and chat template formatting.
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from config import (
    TAR_DIR, POC_CLIPS, INTRINSICS_PATH, ANNOTATE_DIR,
    ANNOTATE_FPS, WINDOW_SIZE, ANNOTATE_MODEL, MAX_NEW_TOKENS,
)
from ingest import load_intrinsics, iter_frames


PROMPT = """You are analyzing consecutive frames (1 FPS) from an egocentric factory video.
The camera is mounted on the worker's head. Respond with ONLY a valid JSON object, no other text:
{
  "active": true or false,
  "object_name": "name of object being manipulated, or null if not active",
  "rigid": true or false or null,
  "bbox": [x1, y1, x2, y2] or null,
  "ecot": {
    "scene": "brief spatial description of the workspace",
    "subtask": "what specific action the hands are performing",
    "motion": "direction and type of hand/object motion",
    "prediction": "what the worker will likely do next"
  },
  "safety_tags": []
}

Rules:
- "active" is true ONLY if hands are visibly manipulating, assembling, or moving an object
- "active" is false if hands are idle, resting, or not visible
- If not active, set object_name, rigid, bbox to null
- bbox is pixel coordinates [x1, y1, x2, y2] of the manipulated object
- "rigid" is true for solid parts/tools, false for cloth/rope/flexible materials
- Respond with ONLY the JSON object, no explanation"""


def load_model():
    # Choice: fp16 on cuda, ~16 GB VRAM for 7B model.
    # Alternative was bf16 — marginal quality difference, but fp16
    # has broader hardware support (3090 handles both, but fp16 is safer).
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ANNOTATE_MODEL, torch_dtype=torch.float16, device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(ANNOTATE_MODEL)
    return model, processor


def to_pil(bgr_frame):
    return Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))


def make_windows(frames_list):
    windows = []
    for i in range(0, len(frames_list), WINDOW_SIZE):
        chunk = frames_list[i:i + WINDOW_SIZE]
        if len(chunk) >= 2:
            windows.append(chunk)
    return windows


def parse_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def annotate_window(window, model, processor):
    content = [{"type": "image", "image": to_pil(f[2])} for f in window]
    content.append({"type": "text", "text": PROMPT})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    out_ids = ids[:, inputs.input_ids.shape[1]:]
    resp = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return parse_response(resp)


def run():
    model, processor = load_model()
    K, D, calib_size = load_intrinsics(INTRINSICS_PATH)
    ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)

    for clip in POC_CLIPS:
        video_path = TAR_DIR / f"{clip}.mp4"
        frames = list(iter_frames(video_path, K, D, calib_size, fps=ANNOTATE_FPS))
        windows = make_windows(frames)
        annotations = []

        for i, window in enumerate(windows):
            ann = annotate_window(window, model, processor)
            ann["window_idx"] = i
            ann["t_start_s"] = round(window[0][1] / 1000.0, 2)
            ann["t_end_s"] = round(window[-1][1] / 1000.0, 2)
            ann["frame_indices"] = [f[0] for f in window]
            annotations.append(ann)
            # Choice: print progress every 10 windows since total runtime ~40-60 min.
            # Alternative was silent — but 72 windows with no output is nerve-wracking.
            if (i + 1) % 10 == 0:
                print(f"  {clip}: {i+1}/{len(windows)} windows")

        out = ANNOTATE_DIR / f"{clip}_annotations.json"
        out.write_text(json.dumps(annotations, indent=2))
        active = sum(1 for a in annotations if a.get("active"))
        print(f"{clip}: {active}/{len(annotations)} active windows")

    del model, processor
    torch.cuda.empty_cache()


run()
