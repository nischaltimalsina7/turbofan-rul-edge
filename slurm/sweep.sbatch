#!/usr/bin/env bash
#SBATCH --job-name=edgecloud-sweep
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8        # 1 (edge) + 2 (edge2) for the caps, the rest for cloud + podman overhead
#SBATCH --mem=16G
#SBATCH --time=04:00:00          # pilot needs minutes; full sweep ~1-1.5 h (see design.md §3a)
#SBATCH --output=results/slurm-%j.out
# No --gres=gpu: the served model runs on CPU in both roles by design.
#
# Submit from the repo root:  sbatch slurm/sweep.sbatch pilot|full
# Watch:                      squeue -u $USER ; tail -f results/slurm-<jobid>.out
set -euo pipefail
MODE=${1:-pilot}
cd "$SLURM_SUBMIT_DIR"
IMG=${IMAGE:-localhost/edgecloud:dev}
NET=${NETWORK:-edgecloud}

# Slurm job steps have no systemd user session; give Podman a runtime dir and
# the cgroupfs manager so it does not look for one. Harmless if already set.
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/podman-run-$(id -u)}; mkdir -p "$XDG_RUNTIME_DIR"
export CONTAINERS_CGROUP_MANAGER=cgroupfs

cleanup() { podman rm -f edge edge2 cloud >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup
podman network exists "$NET" || podman network create "$NET"

# >>> ADAPT the image name, the server command and the client env to yours. <<<
# Keep: the names cloud / edge / edge2, the shared network, --cap-add NET_ADMIN on
# all three, --cpus 1 --memory 1g on edge and --cpus 2 --memory 2g on edge2.
# Those are the experiment.
# Cloud: uncapped (within the Slurm allocation), serves the FP32 model.
podman run -d --name cloud --network "$NET" --cap-add NET_ADMIN \
  -v "$PWD/models:/app/models:ro" \
  "$IMG" python -m src.cloud_server            # <- your server command

# Edge profiles: two emulated gateways, both running so the sweep never restarts
# a container. small = 1 CPU / 1 GiB (edge), medium = 2 CPU / 2 GiB (edge2).
# Both mount the same results/ so runs.csv is one file.
for spec in "edge 1 1g small" "edge2 2 2g medium"; do
  set -- $spec
  podman run -d --name "$1" --network "$NET" --cap-add NET_ADMIN \
    --cpus "$2" --memory "$3" \
    -e SERVER_URL=http://cloud:8000 -e EDGE_PROFILE="$4" \
    -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/results:/app/results" \
    "$IMG" sleep infinity
done

# Wait for the server, then prove the caps and the link before measuring anything.
for i in $(seq 1 30); do
  podman exec edge python -c "import urllib.request;urllib.request.urlopen('http://cloud:8000/health',timeout=2)" 2>/dev/null && break   # <- your health URL
  sleep 1
done
for c in edge edge2; do
  echo "$c cpu.max=$(podman exec $c cat /sys/fs/cgroup/cpu.max) memory.max=$(podman exec $c cat /sys/fs/cgroup/memory.max)"
done
net/netem.sh 50 unlimited && podman exec edge ping -c 3 cloud | tail -1 && podman exec edge2 ping -c 3 cloud | tail -1 && net/netem.sh clear

bash src/run_sweep.sh "$MODE"   # executes docs/doe_cells.csv
