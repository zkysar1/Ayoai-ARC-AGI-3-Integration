#!/bin/bash
# Offline arm runs on the 15 dev games, in parallel processes (g-376-24's launch
# script, generalized for g-376-25). Usage, from anywhere:
#
#   eval/run_arms.sh <out-dir> <run>=<arm> [<run>=<arm> ...]
#   e.g. eval/run_arms.sh /tmp/g37625 Z1=Z Z2=Z N=N G=G P=P B=B
#
# Each run plays every dev game as the named adapters/arc_theory.THEORY_ARMS arm at
# win-test share ${SHARE:-0.5}, in three processes (ft09 alone: an admitted theory there
# runs the planner to its cap on many moves). Each process writes
# <out-dir>/<run>-<a|b|c>.json and .log; the log ends with EXIT=<rc>. Each game is
# recorded under ${RECORDINGS_ROOT:-~/.ayoai-arc/recordings/<out-dir name>}/<run>.
# The model key comes from the environment (ANTHROPIC_API_KEY) and is never logged.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 2 ] || { echo "usage: $0 <out-dir> <run>=<arm> [...]" >&2; exit 2; }
out=$1
shift
mkdir -p "$out"
root=${RECORDINGS_ROOT:-$HOME/.ayoai-arc/recordings/$(basename "$out")}
parts=("ft09" "ar25 bp35 cd82 cn04 ka59 lp85 ls20" "r11l re86 sp80 su15 tn36 vc33 wa30")
names=(a b c)
for spec in "$@"; do
  run=${spec%%=*}
  arm=${spec#*=}
  for i in 0 1 2; do
    stem="$out/$run-${names[$i]}"
    # shellcheck disable=SC2086  # the game list is meant to split into words
    ( RECORDINGS_DIR="$root/$run" .venv/bin/python eval/adapter_run.py --player port \
        --theory-share "${SHARE:-0.5}" --theory-arm "$arm" --record --games ${parts[$i]} \
        --out "$stem.json" > "$stem.log" 2>&1
      echo "EXIT=$?" >> "$stem.log" ) &
  done
done
wait
echo "ALL-ARM-RUNS-DONE"
