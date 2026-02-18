import json
import re
import cv2
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

import config
from parse import fmt, parse_time

PROMPT = """This {dur}s clip is from a head-mounted camera on a factory worker. Each frame = 1 second.

The worker's task cycle (when on-task):
1. Pick up a metal bar from the right-side box
2. Place it into the press machine
3. Places their hands on the side of the press machine
4. Remove it after pressing
5. Place it into the left-side box

For each second, classify: is the worker performing ANY step of this cycle?

CYCLE = actively doing step 1, 2, 3, or 4
OTHER = anything else (walking, idle, talking, carrying items, adjusting, leaving station)

List status at the start of the clip, then ONLY when it changes.
Format (nothing else):

{start} CYCLE
MM:SS OTHER reason
MM:SS CYCLE

If on-task the whole time, just: {start} CYCLE"""


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL, dtype=torch.float16, device_map="cuda",
    )
    proc = AutoProcessor.from_pretrained(config.MODEL)
    return model, proc


def extract_window(cap, start_sec, dur, video_fps):
    frames = []
    for t in range(start_sec, start_sec + dur):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * video_fps))
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    return frames


def run_window(frames, start_sec, dur, model, proc):
    prompt = PROMPT.format(dur=dur, start=fmt(start_sec))
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames, "fps": float(config.SAMPLE_FPS)},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        img_in, vid_in, vid_kw = process_vision_info(messages, return_video_kwargs=True)
    except TypeError:
        img_in, vid_in = process_vision_info(messages)
        vid_kw = {}
    if "fps" in vid_kw and isinstance(vid_kw["fps"], list):
        vid_kw["fps"] = vid_kw["fps"][0]

    inputs = proc(
        text=[text], images=img_in, videos=vid_in,
        padding=True, return_tensors="pt", **vid_kw,
    ).to("cuda")
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=config.MAX_TOKENS)
    out = ids[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(out, skip_special_tokens=True)[0]


def parse_transitions(text, start_sec, dur):
    transitions = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d+:\d{2})\s+(CYCLE|OTHER|C|O)\s*(.*)", line, re.IGNORECASE)
        if not m:
            continue
        sec = parse_time(m.group(1))
        if sec is None:
            continue
        raw_status = m.group(2).upper()
        status = "CYCLE" if raw_status in ("C", "CYCLE") else "OTHER"
        reason = m.group(3).strip()
        transitions.append({"sec": sec, "status": status, "reason": reason})

    if not transitions:
        return []

    end_sec = start_sec + dur
    labels = []
    for i, tr in enumerate(transitions):
        next_sec = transitions[i + 1]["sec"] if i + 1 < len(transitions) else end_sec
        for s in range(tr["sec"], min(next_sec, end_sec)):
            labels.append({"sec": s, "time": fmt(s), "status": tr["status"], "reason": tr["reason"]})
    return labels


if not config.CLIPS:
    print(f"no rectified clips in {config.CLIP_DIR}, run ingest first")
    exit(1)

model, proc = load_model()
config.OUT_DIR.mkdir(parents=True, exist_ok=True)

for clip_path in config.CLIPS:
    cap = cv2.VideoCapture(str(clip_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps)
    n_windows = (total_sec - 1) // config.STRIDE_SEC + 1
    print(f"{clip_path.name}: {total_sec}s, {n_windows} windows")

    all_labels = []
    all_raw = []
    starts = range(0, total_sec, config.STRIDE_SEC)

    for start in tqdm(starts, desc=clip_path.stem):
        dur = min(config.WINDOW_SEC, total_sec - start)
        if dur < 3:
            break
        frames = extract_window(cap, start, dur, video_fps)
        if len(frames) < 3:
            break

        raw = run_window(frames, start, dur, model, proc)
        all_raw.append({"window_start": start, "raw": raw})
        labels = parse_transitions(raw, start, dur)
        all_labels.extend(labels)

    cap.release()

    raw_path = config.OUT_DIR / f"{clip_path.stem}_raw.json"
    raw_path.write_text(json.dumps(all_raw, indent=2))

    label_path = config.OUT_DIR / f"{clip_path.stem}_labels.json"
    label_path.write_text(json.dumps(all_labels, indent=2))

    n_cycle = sum(1 for l in all_labels if l["status"] == "CYCLE")
    n_other = sum(1 for l in all_labels if l["status"] == "OTHER")
    print(f"  {n_cycle}s cycle, {n_other}s other, {len(all_labels)}s total")

del model, proc
torch.cuda.empty_cache()