#!/usr/bin/env bash
# Snapshot the machine state around one replication, independent of whatever
# the edge client writes. Appends one row to results/rep_probe.csv.
#
#   src/rep_probe.sh start|end <cell_id> <config> <rtt_ms> <bw> <precision> <edge_profile> <edge_container> <rep>
#
# Reads, from INSIDE the edge container (podman exec):
#   /proc/loadavg            host-wide 1- and 5-minute load (is the machine quiet?)
#   /sys/fs/cgroup/cpu.stat  usage_usec, nr_periods, nr_throttled, throttled_usec
#                            (did the 1-CPU cap actually bite?)
#   /sys/fs/cgroup/cpu.max   must read "100000 100000" for a 1-CPU cap
#   /sys/fs/cgroup/memory.peak
# plus the wall-clock timestamp (ns). start/end pairs are differenced later:
# cpu_busy = Δusage_usec / Δwall ; throttled share = Δnr_throttled / Δnr_periods.
set -uo pipefail
PHASE=${1:?start|end}; CELL=${2:?}; CFG=${3:?}; RTT=${4:?}; BW=${5:?}; PREC=${6:?}; PROF=${7:?}; EDGE=${8:?}; REP=${9:?}
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT="$HERE/results/rep_probe.csv"
mkdir -p "$HERE/results"
[[ -f "$OUT" ]] || echo "run_id,cell_id,config,rtt_ms,bw,precision,edge_profile,edge_container,rep,phase,ts_utc,wall_ns,loadavg1,loadavg5,cg_usage_usec,cg_nr_periods,cg_nr_throttled,cg_throttled_usec,cg_cpu_max,cg_memory_peak_bytes,other_slurm_jobs" > "$OUT"

read -r L1 L5 _ < <(podman exec "$EDGE" cat /proc/loadavg 2>/dev/null || echo "nan nan")
CPUSTAT=$(podman exec "$EDGE" cat /sys/fs/cgroup/cpu.stat 2>/dev/null || true)
val() { echo "$CPUSTAT" | awk -v k="$1" '$1==k{print $2}'; }
CPUMAX=$(podman exec "$EDGE" cat /sys/fs/cgroup/cpu.max 2>/dev/null | tr ' ' '/' || echo n/a)
MEMPEAK=$(podman exec "$EDGE" cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo "")
# other people's jobs on the node right now (ours excluded): the contamination flag
OTHERS=$(squeue -h -o "%u" 2>/dev/null | grep -vc "^$USER$" || echo 0)

echo "${RUN_ID:-none},$CELL,$CFG,$RTT,$BW,$PREC,$PROF,$EDGE,$REP,$PHASE,$(date -u +%Y-%m-%dT%H:%M:%SZ),$(date +%s%N),$L1,$L5,$(val usage_usec),$(val nr_periods),$(val nr_throttled),$(val throttled_usec),$CPUMAX,$MEMPEAK,$OTHERS" >> "$OUT"
