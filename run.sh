cd "$(dirname "$0")"
python ingest.py "${1:-all}"
python VJEPA/encode.py
python VJEPA/detect.py
python VJEPA/analyze.py
