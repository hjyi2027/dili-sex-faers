#!/bin/bash
# Reduce every downloaded quarter that has not been reduced yet.
cd "$(dirname "$0")/.."
for z in data/faers_ascii_*.zip; do
  q=$(basename "$z" .zip | sed 's/faers_ascii_//')
  if [ ! -f "out/q/$q/qc.json" ]; then
    python3 -c "import zipfile,sys; sys.exit(0 if zipfile.ZipFile('$z').testzip() is None else 1)" 2>/dev/null || continue
    python3 scripts/01_reduce.py "$z" out/q >> out/reduce.log 2>&1 || echo "FAILED $q" >> out/reduce.log
  fi
done
