import re


def fmt(sec):
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def parse_time(s):
    parts = s.split(":")
    if len(parts) != 2:
        return None
    return int(parts[0]) * 60 + int(parts[1])


def parse_transitions(text, start_sec, dur):
    transitions = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d+:\d{2})\s+(W|O)\s*(.*)", line, re.IGNORECASE)
        if not m:
            continue
        sec = parse_time(m.group(1))
        if sec is None:
            continue
        status = m.group(2).upper()
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
