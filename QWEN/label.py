import json
import cv2
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

import config
from parse import fmt, parse_transitions

SYSTEM = (
    "You analyze egocentric factory video. The worker's task: pick up a small electronic "
    "component, pick up a white ring, fit the ring onto the component, place it down. "
    "This repeats continuously on a green conveyor belt."
)

PROMPT = """This {dur}s clip starts at {start}. Each frame = 1 second.
List the worker's status at the start, then ONLY when it changes.
Use exactly this format, nothing else:

{start} W
MM:SS O reason
MM:SS W

W = doing assembly. O = anything else (talking, looking away, idle, adjusting).
If working the whole time, just write one line: {start} W"""


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
        {"role": "system", "content": SYSTEM},
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


model, proc = load_model()
config.OUT_DIR.mkdir(parents=True, exist_ok=True)

for clip_path in tqdm(config.CLIPS, desc="clips", unit="clip"):
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

    n_work = sum(1 for l in all_labels if l["status"] == "W")
    n_off = sum(1 for l in all_labels if l["status"] == "O")
    print(f"  {n_work}s work, {n_off}s off-task, {len(all_labels)}s total")

del model, proc
torch.cuda.empty_cache()
