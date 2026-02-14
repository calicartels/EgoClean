#!/bin/bash
cd "$(dirname "$0")"
case "${1:-download}" in
  download) echo "Downloading part000.tar and intrinsics.json from Egocentric-100K"; python -m ingest.download ;;
  *) echo "Unknown: $1"; exit 1 ;;
esac
