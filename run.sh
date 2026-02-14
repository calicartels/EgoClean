#!/bin/bash
cd "$(dirname "$0")"
case "${1:-all}" in
  download)      python ingest.py download ;;
  extract)       python ingest.py extract ;;
  check-rectify) python ingest.py check-rectify ;;
  triage)        python -m triage.detect && python -m triage.merge ;;
  all)
    python ingest.py download
    python ingest.py extract
    python ingest.py check-rectify
    python -m triage.detect
    python -m triage.merge
    ;;
  *) echo "unknown: $1"; exit 1 ;;
esac
