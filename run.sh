#!/bin/bash
cd "$(dirname "$0")"
case "${1:-all}" in
  download)      python ingest.py download ;;
  extract)       python ingest.py extract ;;
  check-rectify) python ingest.py check-rectify ;;
  all)
    python ingest.py download
    python ingest.py extract
    python ingest.py check-rectify
    ;;
  *) echo "unknown: $1"; exit 1 ;;
esac
