#!/bin/bash
# Keep reducing until the downloader has exited and every valid zip is reduced.
cd "$(dirname "$0")/.."
while true; do
  ./scripts/run_reduce.sh
  if ! pgrep -f 00_download.py > /dev/null; then ./scripts/run_reduce.sh; break; fi
  sleep 30
done
echo ALLDONE >> out/reduce.log
