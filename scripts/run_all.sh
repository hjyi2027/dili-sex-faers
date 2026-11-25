#!/bin/bash
# Full pipeline from the downloaded zips into a fresh output root.
# Usage: scripts/run_all.sh <output_root>   (run from the project directory)
set -e
ROOT=$1
mkdir -p "$ROOT/q"
for z in data/faers_ascii_*.zip; do
  python3 scripts/01_reduce.py "$z" "$ROOT/q"
  python3 scripts/01b_indication.py "$z" "$ROOT/q"
done
python3 scripts/02_merge_dedup.py "$ROOT/q" "$ROOT/merged"
bash scripts/run_from_build.sh "$ROOT"
echo CLEAN_RUN_DONE
