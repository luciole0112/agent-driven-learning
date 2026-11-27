#!/usr/bin/env bash
# Step 0 comparison grid (docs/cifar_mbv2_design_v1.md, section 7).
# Run inside the adl-gpu container:  bash scripts/run_step0_grid.sh <budget> <seeds...>
# Each run writes runs/mbv2/baselines/<start>_<method>_<space>_s<seed>_b<budget>.json
set -u
BUDGET=${1:-2000000}
shift || true
SEEDS=${@:-0}
OUT=runs/mbv2/baselines
mkdir -p "$OUT" runs/mbv2/logs
run() {  # start method space seed extra...
  local s=$1 me=$2 sp=$3 sd=$4; shift 4
  local f="$OUT/${s}_${me}_${sp//[:,]/}_s${sd}_b${BUDGET}.json"
  [ -f "$f" ] && { echo "skip $f"; return; }
  echo "=== $s $me $sp seed=$sd $(date +%T)"
  python scripts/run_mbv2_baseline.py --start runs/mbv2/$s.pt --method $me --space $sp \
    --budget-images $BUDGET --seed $sd --out "$f" "$@" | grep -E '"acc_(dev|heldout|shift)_(initial|final)"|generations|steps|accepted' | tr -d '\n'
  echo
}
for sd in $SEEDS; do
  # Model 1: S1 (frozen ImageNet trunk + linear probe head)
  run S1 es head $sd
  run S1 random head $sd
  run S1 backprop head $sd --lr 0.01
  # Model 2: R0c (random init, BN re-estimated)
  run R0c es stem $sd --recal-every 50
  run R0c random stem $sd
  run R0c backprop stem $sd --bn-train
  run R0c es head $sd
  run R0c backprop all $sd --bn-train
done
# Fixed rules (deterministic, no budget)
for s in S1 R0c; do run $s ncm head 0; run $s ridge head 0; done
