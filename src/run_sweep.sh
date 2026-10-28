#!/usr/bin/env bash
# Executes the designed-experiment table docs/doe_cells.csv, row by row in
# run_order. The table is the experiment: edit src/make_doe.py, regenerate,
# commit; never edit the loop here to "just add a cell".
#
#   src/run_sweep.sh pilot   -> the 3 pilot rows (cloud-only, RTT 0/50/200), 1 rep, 20 engines
#   src/run_sweep.sh pilot2  -> one cell per placement x edge profile at RTT 50 / unlimited /
#                               float32 (5 cells), 1 rep, 20 engines: proves every path works
#   src/run_sweep.sh full    -> all 77 rows, reps/engines from the table
#   DRY_RUN=1 src/run_sweep.sh full   -> print the commands, run nothing
#
# Assumes `edge`, `edge2` and `cloud` containers are running (slurm/sweep.sbatch).
# One sweep at a time on the workstation: `squeue -u $USER`, `podman ps`.
set -euo pipefail
MODE=${1:-pilot}
HERE=$(cd "$(dirname "$0")/.." && pwd)
TABLE="$HERE/docs/doe_cells.csv"
export RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_${MODE}}
[[ "$MODE" == "pilot" || "$MODE" == "pilot2" || "$MODE" == "preflight" || "$MODE" == "full" ]] || { echo "mode must be pilot|pilot2|preflight|full"; exit 2; }
[[ -f "$TABLE" ]] || { echo "missing $TABLE — run: python src/make_doe.py"; exit 2; }

run() { if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "+ $*"; else "$@"; fi; }

# >>> THE ONE LINE TO ADAPT TO YOUR CLIENT <<<
# It must accept: --config cloud|hybrid|edge  --rtt-ms N  --bw X  --precision float32|float16
#                 --edge-profile small|medium  --rep N  --run-id ID  [--limit N]
# and append one row per inference to results/runs.csv (columns: see collect.txt).
# $EDGE is the container for this cell's edge profile (edge = 1 CPU/1 GiB, edge2 = 2 CPU/2 GiB).
client() {
  local CPUSET
  if [[ "$EDGE" == "edge" ]]; then
    CPUSET="0,16"
  else
    CPUSET="0,1,16,17"
  fi
  run podman exec -e RUN_ID="$RUN_ID" "$EDGE" taskset -c "$CPUSET" python -m src.edge_client "$@"
}
probe()  { run "$HERE/src/rep_probe.sh" "$@"; }   # machine-state snapshot, see rep_probe.sh

# Provenance: without this block no number is reproducible.
mkdir -p "$HERE/results"
META="$HERE/results/meta_${RUN_ID}.txt"
if [[ "${DRY_RUN:-0}" != "1" ]]; then {
  echo "run_id=$RUN_ID"; echo "mode=$MODE"; echo "host=$(hostname)"
  echo "table_sha256=$(sha256sum "$TABLE" | cut -c1-16)"
  echo "git_commit=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo untracked)"
  echo "git_dirty=$(git -C "$HERE" status --porcelain 2>/dev/null | wc -l)"
  echo "podman=$(podman --version)"; echo "image=$(podman inspect -f '{{.ImageName}} {{.Image}}' edge)"
  echo "slurm_job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
  echo "load_at_start=$(cut -d' ' -f1-3 /proc/loadavg)"
  for c in ${EDGE_CONTAINERS:-edge edge2}; do
    echo "${c}_cpu_max=$(podman exec "$c" cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
    echo "${c}_mem_max=$(podman exec "$c" cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"
  done
} > "$META"; fi

# Read the table with python (robust CSV parsing), emit one line per cell:
# cell_id placement rtt bw precision edge_profile edge_container reps engines pilot
python3 - "$TABLE" <<'EOF' | while read -r CELL PLACE RTT BW PREC PROF EDGE REPS ENG PILOT; do
import csv, sys
for r in sorted(csv.DictReader(open(sys.argv[1])), key=lambda r: int(r["run_order"])):
    print(r["cell_id"], r["placement"], r["rtt_ms"], r["bw"], r["precision"],
          r["edge_profile"], r["edge_container"], r["reps"], r["engines_per_rep"], r["pilot"])
EOF
  if [[ "$MODE" == "preflight" ]]; then
    [[ "$CELL" == "C35" ]] || continue
    REPS=1; LIMIT=(--limit 20)
  elif [[ "$MODE" == "pilot" ]]; then
    [[ "$PILOT" == "True" ]] || continue
    REPS=1; LIMIT=(--limit 20)
  elif [[ "$MODE" == "pilot2" ]]; then
    [[ "$PLACE" == "edge" || ( "$RTT" == "50" && "$BW" == "unlimited" && "$PREC" == "float32" ) ]] || continue
    REPS=1; LIMIT=(--limit 20)
  else
    LIMIT=()
  fi
  echo "== $CELL $PLACE rtt=$RTT bw=$BW prec=$PREC profile=$PROF($EDGE) reps=$REPS"
  if [[ "$PLACE" == "edge" ]]; then
    run "$HERE/net/netem.sh" clear
    for rep in $(seq 0 $((REPS-1))); do
      probe start "$CELL" edge 0 none int8 "$PROF" "$EDGE" "$rep"
      client --config edge --rtt-ms 0 --bw none --precision int8 --edge-profile "$PROF" --rep "$rep" --run-id "$RUN_ID" "${LIMIT[@]}"
      probe end   "$CELL" edge 0 none int8 "$PROF" "$EDGE" "$rep"
    done
  else
    run "$HERE/net/netem.sh" "$RTT" "$BW"
    [[ "${DRY_RUN:-0}" == "1" ]] || "$HERE/net/netem.sh" show >> "$HERE/results/netem_log_${RUN_ID}.txt" 2>/dev/null || true
    for rep in $(seq 0 $((REPS-1))); do
      probe start "$CELL" "$PLACE" "$RTT" "$BW" "$PREC" "$PROF" "$EDGE" "$rep"
      client --config "$PLACE" --rtt-ms "$RTT" --bw "$BW" --precision "$PREC" --edge-profile "$PROF" --rep "$rep" --run-id "$RUN_ID" "${LIMIT[@]}"
      probe end   "$CELL" "$PLACE" "$RTT" "$BW" "$PREC" "$PROF" "$EDGE" "$rep"
    done
  fi
done

run "$HERE/net/netem.sh" clear
[[ "${DRY_RUN:-0}" == "1" ]] || echo "load_at_end=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$META"
echo "sweep $RUN_ID complete. Commit: results/runs.csv results/rep_probe.csv results/meta_${RUN_ID}.txt results/netem_log_${RUN_ID}.txt ; then git tag"
