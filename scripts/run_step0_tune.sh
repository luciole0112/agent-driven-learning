#!/usr/bin/env bash
# Step 0: step-size sweep for the zero-order baselines (ES, random search).
# The default sigma 0.01 / lr 0.02 made ES and random search *lower* S1's accuracy
# (seed 0, 2M images), so they were too weak to be fair comparisons. The setting is
# chosen on dev at a 500k budget (heldout is recorded but not used for the choice).
# Run inside the adl-gpu container:  bash scripts/run_step0_tune.sh
set -u
B=${1:-500000}
OUT=runs/mbv2/tune
mkdir -p "$OUT"
run() {  # start method space tag extra...
  local s=$1 me=$2 sp=$3 tag=$4; shift 4
  local f="$OUT/${s}_${me}_${sp}_${tag}_b${B}.json"
  [ -f "$f" ] && { echo "skip $f"; return; }
  echo "=== $s $me $sp $tag $(date +%T)"
  python scripts/run_mbv2_baseline.py --start runs/mbv2/$s.pt --method $me --space $sp \
    --budget-images $B --seed 0 --out "$f" "$@" | grep -E '"acc_(dev|heldout)_final"' | tr -d '\n'
  echo
}
run S1 es head s1e-3_lr1e-3 --sigma 0.001 --lr 0.001
run S1 es head s1e-3_lr3e-4 --sigma 0.001 --lr 0.0003
run S1 es head s3e-3_lr3e-3 --sigma 0.003 --lr 0.003
run S1 es head s3e-4_lr1e-4 --sigma 0.0003 --lr 0.0001
run S1 random head s1e-3 --sigma 0.001
run S1 random head s3e-4 --sigma 0.0003
run R0c es head s1e-2_lr3e-3 --sigma 0.01 --lr 0.003
run R0c es head s3e-2_lr2e-2 --sigma 0.03 --lr 0.02
run R0c es stem s3e-2_lr2e-2 --sigma 0.03 --lr 0.02 --recal-every 50
