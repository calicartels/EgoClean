import json
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


SYSTEM = """You are analyzing an egocentric factory video sampled at 1 frame per second.
Frame numbers correspond to seconds (frame 0 = 0:00, frame 60 = 1:00, frame 120 = 2:00).

The worker's assigned task is: attaching small white rings (plastic/rubber gaskets) onto
small rectangular electronic components on a green conveyor belt. The process is: pick up
component with one hand, pick up white ring with the other, fit ring around circular feature
on component, place finished item down. This is repeated continuously."""

QUERY = """Identify ALL moments where the worker is NOT performing the assigned task.
This includes: turning head to look at or talk to a coworker, idling, adjusting posture
without working, looking away from the workstation, or any other non-assembly activity.

For each interruption, give me:
- The start and end time in MM:SS format
- What the worker is doing instead

Then list the time ranges where the worker IS performing the assembly task.

Respond with ONLY a JSON object:
{"off_task": [{"start": "MM:SS", "end": "MM:SS", "reason": "what they're doing instead"}], "on_task": [{"start": "MM:SS", "end": "MM:SS"}]}"""


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ANNOTATE_MODEL, torch_dtype=torch.float16, device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(ANNOTATE_MODEL)
    return model, processor


def to_pil(bgr_frame):
    return Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))


def collect_frames(video_path, K, D, calib_size):
    return [to_pil(f) for _, _, f in iter_frames(video_path, K, D, calib_size, fps=ANNOTATE_FPS)]


def parse_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def annotate_clip(frames, model, processor):
    content = [
        {"type": "video", "video": frames, "sample_fps": 1.0},
        {"type": "text", "text": QUERY},
    ]
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
        print(f"  raw response saved")

        annotations = parse_response(raw_resp)
        out = ANNOTATE_DIR / f"{clip}_annotations.json"
        out.write_text(json.dumps(annotations, indent=2))

        n_off = len(annotations.get("off_task", []))
        n_on = len(annotations.get("on_task", []))
        print(f"{clip}: {n_off} off-task periods, {n_on} on-task periods")

    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
