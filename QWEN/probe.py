import json
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import config
from parse import fmt

# Known windows to probe — mix of confirmed work and confirmed anomaly.
# Each is (clip_index, start_sec, dur_sec, ground_truth_label).
PROBES = [
    # Clip 1: pure work
    (1, 60, 10, "work"),
    (1, 150, 10, "work"),
    # Clip 1: anomaly at 03:28-04:22 (material handling / not assembling)
    (1, 208, 10, "anomaly_material_handling"),
    (1, 230, 10, "anomaly_material_handling"),
    # Clip 1: anomaly at 07:00-07:13 (ending work)
    (1, 420, 13, "anomaly_ending"),
    # Clip 2: pure work
    (2, 300, 10, "work"),
    (2, 600, 10, "work"),
    # Clip 2: anomaly at 02:07-02:54 (not working)
    (2, 127, 10, "anomaly_not_working"),
    (2, 150, 10, "anomaly_not_working"),
    # Clip 2: anomaly at 16:09-17:03 (gets up)
    (2, 969, 10, "anomaly_gets_up"),
    (2, 990, 10, "anomaly_gets_up"),
]

# Choice: completely open-ended prompt. No mention of assembly, no task description,
# no W/O format. Just ask what the person is doing. If the model describes different
# activities for work vs anomaly windows, it CAN see the difference — the previous
# prompt was suppressing it. If it says the same thing for all, 7B can't distinguish.
PROMPT = "Describe what the person is doing in this clip. Be specific about their hands, posture, and what they're interacting with."


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL, dtype=torch.float16, device_map="cuda",
    )
    proc = AutoProcessor.from_pretrained(config.MODEL)
    return model, proc


def extract_window(clip_idx, start_sec, dur):
    path = config.CLIP_DIR / f"rectified_clip_{clip_idx}.mp4"
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    # Choice: sample at 1fps (same as label.py) for consistency.
    for t in range(start_sec, start_sec + dur):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def run_probe(frames, model, proc):
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames, "fps": 1.0},
            {"type": "text", "text": PROMPT},
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

    n_tok = inputs.input_ids.shape[1]

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=256)
    out = ids[:, inputs.input_ids.shape[1]:]
    resp = proc.batch_decode(out, skip_special_tokens=True)[0]
    return resp, n_tok


model, proc = load_model()
results = []

for clip_idx, start, dur, gt in tqdm(PROBES, desc="probe"):
    tag = f"clip{clip_idx} {fmt(start)}-{fmt(start+dur)} [{gt}]"
    print(f"\n{'='*60}")
    print(f"PROBE: {tag}")
    print(f"{'='*60}")

    frames = extract_window(clip_idx, start, dur)
    resp, n_tok = run_probe(frames, model, proc)

    print(f"  frames: {len(frames)}, input tokens: {n_tok}")
    print(f"  response: {resp[:300]}")

    results.append({
        "clip": clip_idx,
        "start": start,
        "dur": dur,
        "start_fmt": fmt(start),
        "ground_truth": gt,
        "n_frames": len(frames),
        "n_tokens": n_tok,
        "response": resp,
    })

config.OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = config.OUT_DIR / "probe_results.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nsaved {out_path}")

del model, proc
torch.cuda.empty_cache()