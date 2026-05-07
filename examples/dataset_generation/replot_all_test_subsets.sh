#!/bin/bash
# Re-run 200-sample QA subsets for every layer + replot all 4 props
# (facies, poro, perm, facies_alluvsim) on each. Use this to refresh
# $SCRATCH/resmill_dataset/<layer>_test/ after touching
# any layer/sampler code or any config.
#
# Skips a preset if its output dir already has 200 fresh shards (so
# you can resume after a disconnect by just re-running this script).
# NOTE: no `set -e` — the piped commands below tolerate empty greps and
# we want to keep going on a single-preset failure.

PY=$WORK/conda_envs/resmill/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=$SCRATCH/resmill_dataset
RUNNER="$HERE/run_dataset.py"
PLOTTER="$HERE/plot_dataset.py"
STATS_PLOTTER="$HERE/plot_dataset_stats.py"

declare -A presets=(
  [lobes]="lobes_test"
  [pv_shoestring]="channels_pv_test"
  [cb_labyrinth]="channels_cb_lab_test"
  [sh_distal]="sh_distal_test"
  [sh_proximal]="sh_proximal_test"
  [meander_oxbow]="oxbow_test"
  [cb_jigsaw]="cb_jigsaw_test"
  [delta]="delta_test"
)

LIMIT=100   # samples per preset for the QA subset

for preset in lobes pv_shoestring cb_labyrinth sh_distal sh_proximal meander_oxbow cb_jigsaw delta; do
  outdir="$ROOT/${presets[$preset]}"
  cfg="$HERE/config_full_${preset}.json"
  echo "=========================================================="
  echo "=== $preset"
  echo "=========================================================="

  # Skip if all 4 picture-subdirs already have $LIMIT PNGs each (means
  # the regen+replot completed on a prior run).
  is_complete=1
  for sub in facies_binary_pictures poro_pictures perm_pictures facies_alluvsim_pictures; do
    n=$(ls "$outdir/$sub" 2>/dev/null | wc -l)
    if [ "$n" -ne "$LIMIT" ]; then is_complete=0; break; fi
  done
  if [ "$is_complete" -eq 1 ]; then echo "  (skip — already complete)"; continue; fi

  rm -rf "$outdir"
  $PY "$RUNNER" "$cfg" --workers 32 --limit "$LIMIT" --output-dir "$outdir" \
    | grep -E "Done|per-job|failed:" | tail -3
  $PY "$PLOTTER" "$outdir" --workers 32 | tail -3
  $PY "$STATS_PLOTTER" "$outdir" 2>&1 | tail -2
  echo
done
echo "=== ALL REGEN+REPLOT DONE ==="
