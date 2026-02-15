import json
import re
import sys

import cv2
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from config import (
    TAR_DIR, POC_CLIPS, INTRINSICS_PATH, ANNOTATE_DIR,
    ANNOTATE_FPS, ANNOTATE_MODEL, MAX_NEW_TOKENS,
)
from ingest import load_intrinsics, iter_frames


PROMPT = """You are analyzing an egocentric factory video sampled at 1 frame per second.
The camera is mounted on the worker's head. Frame numbers correspond to seconds.

Identify ALL distinct manipulation actions in this video. For each action, return a JSON
object in an array. Respond with ONLY a valid JSON array, no other text:

[
  {
    "action_id": 1,
    "t_start_s": <first frame number where this action begins>,
    "t_end_s": <last frame number where this action ends>,
    "active": true,
    "objects": [
      {"object_name": "name or null", "bbox": [x1, y1, x2, y2], "rigid": true}
    ],
    "ecot": {
      "scene": "brief spatial description of the workspace",
      "subtask": "what specific action the hands are performing",
      "motion": "direction and type of hand/object motion",
      "prediction": "what the worker will likely do next"
    }
  }
]

Rules:
- Cover the ENTIRE video timeline — if a period has no manipulation, include:
  {"action_id": N, "t_start_s": X, "t_end_s": Y, "active": false, "objects": [], "ecot": null}
- "objects" is an array: empty for idle, one entry for single-object actions,
  multiple entries for multi-object actions (e.g. assembling part A into part B)
- object_name can be null if the object is not identifiable
- bbox is pixel coordinates [x1, y1, x2, y2] of each object in the MIDDLE frame
- "rigid" is true for solid parts/tools, false for cloth/rope/flexible materials
- Do not merge different actions — if the worker switches objects, that is a new segment
- Respond with ONLY the JSON array"""


def load_model():
    # Choice: fp16, not bf16. Both work on 3090 but fp16 is the safer default
    # across GPU variants. bf16 has slight numerical advantage but no practical
    # difference at inference time for a 7B model.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ANNOTATE_MODEL, torch_dtype=torch.float16, device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(ANNOTATE_MODEL)
    return model, processor


def to_pil(bgr_frame):
    return Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))


def collect_frames(video_path, K, D, calib_size):
    return [to_pil(f) for _, _, f in iter_frames(video_path, K, D, calib_size, fps=ANNOTATE_FPS)]


def _close_truncated_json(text):
    """Attempt to close a truncated JSON array so we can salvage complete segments."""
    # Strip any trailing incomplete key-value or comma
    text = re.sub(r",\s*$", "", text.rstrip())
    # Close any open braces/brackets
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    # Walk back to the last complete object — find last '}' that is part of the array
    # and trim everything after it, then close the array.
    last_brace = text.rfind("}")
    if last_brace != -1:
        text = text[:last_brace + 1]
    # Recount what needs closing
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    text += "".join(reversed(stack))
    return text


def parse_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if "```" in text:
        text = text.split("```")[0]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    def try_parse(s):
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return json.loads(s)

    try:
        return try_parse(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                return try_parse(match.group(0))
            except json.JSONDecodeError:
                pass
        # Truncation recovery: try closing brackets to salvage complete segments
        try:
            repaired = _close_truncated_json(text)
            result = try_parse(repaired)
            print("WARNING: model output was truncated; salvaged"
                  f" {len(result)} complete segments", file=sys.stderr)
            return result
        except (json.JSONDecodeError, Exception):
            pass
        raise


def annotate_clip(frames, model, processor):
    # Choice: video with list of PIL frames + fps key for frame-list input.
    # Alternative was passing a file path — but that skips fisheye rectification.
    content = [
        {"type": "video", "video": frames, "fps": 1.0},
        {"type": "text", "text": PROMPT},
    ]
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Choice: return_video_kwargs=True to get proper video processing params.
    # Older qwen-vl-utils may not support this — fallback to basic call.
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True,
        )
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}

    if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
        video_kwargs = {**video_kwargs, "fps": video_kwargs["fps"][0]}

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to("cuda")

    n_tokens = inputs.input_ids.shape[1]
    print(f"  input tokens: {n_tokens}")

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    out_ids = ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0]


def run():
    model, processor = load_model()
    K, D, calib_size = load_intrinsics(INTRINSICS_PATH)
    ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)

    for clip in POC_CLIPS:
        video_path = TAR_DIR / f"{clip}.mp4"
        if not video_path.exists():
            print(f"skip {clip}: not found", file=sys.stderr)
            continue
        print(f"collecting rectified frames for {clip}...")
        frames = collect_frames(video_path, K, D, calib_size)
        print(f"  {len(frames)} frames, sending as video to Qwen2.5-VL...")

        raw_resp = annotate_clip(frames, model, processor)
        (ANNOTATE_DIR / f"{clip}_raw_response.txt").write_text(raw_resp)

        annotations = parse_response(raw_resp)
        out = ANNOTATE_DIR / f"{clip}_annotations.json"
        out.write_text(json.dumps(annotations, indent=2))

        active = sum(1 for a in annotations if a.get("active"))
        print(f"{clip}: {active}/{len(annotations)} active segments")

    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()