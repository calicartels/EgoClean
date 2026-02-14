import json

from config import POC_CLIPS, TRIAGE_DIR, IOU_THRESHOLD, MERGE_GAP_S, MIN_DURATION_S


def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def flag_active(detections):
    for det in detections:
        hands = [b for b, l in zip(det["boxes"], det["labels"]) if "hand" in str(l).lower()]
        objects = [b for b, l in zip(det["boxes"], det["labels"]) if "object" in str(l).lower()]
        active = False
        for hb in hands:
            for ob in objects:
                if compute_iou(hb, ob) > IOU_THRESHOLD:
                    active = True
                    break
            if active:
                break
        det["active"] = active
    return detections


def merge_segments(detections):
    active_times = sorted([d["timestamp_ms"] / 1000.0 for d in detections if d.get("active")])
    if not active_times:
        return []
    segments = []
    start = active_times[0]
    end = active_times[0]
    for t in active_times[1:]:
        if t - end <= MERGE_GAP_S:
            end = t
        else:
            if end - start >= MIN_DURATION_S:
                segments.append({"start_s": round(start, 2), "end_s": round(end, 2)})
            start = t
            end = t
    if end - start >= MIN_DURATION_S:
        segments.append({"start_s": round(start, 2), "end_s": round(end, 2)})
    return segments


def run():
    for clip in POC_CLIPS:
        det_path = TRIAGE_DIR / f"{clip}_detections.json"
        if not det_path.exists():
            print(f"skip {clip}: no detections", file=__import__("sys").stderr)
            continue
        detections = json.loads(det_path.read_text())
        detections = flag_active(detections)
        segments = merge_segments(detections)

        manifest = {
            "clip": clip,
            "total_frames": len(detections),
            "active_frames": sum(1 for d in detections if d.get("active")),
            "segments": segments,
        }
        (TRIAGE_DIR / f"{clip}_manifest.json").write_text(json.dumps(manifest, indent=2))
        det_path.write_text(json.dumps(detections))

        pct = manifest["active_frames"] / manifest["total_frames"] * 100 if manifest["total_frames"] else 0
        print(f"{clip}: {manifest['active_frames']}/{manifest['total_frames']} active ({pct:.0f}%), {len(segments)} segments")


if __name__ == "__main__":
    run()
