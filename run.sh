cd "$(dirname "$0")"
python ingest.py "${1:-all}"
