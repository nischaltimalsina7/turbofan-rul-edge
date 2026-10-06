#!/usr/bin/env bash
# Week-6 checks. Run on the workstation inside a short Slurm allocation:
#   srun --cpus-per-task=2 --mem=4G --time=00:15:00 bash net/preflight.sh
# Each check prints PASS/FAIL and what a FAIL means. Nothing here needs sudo.
set -uo pipefail
IMG=${IMAGE:-localhost/edgecloud:dev}
NET=${NETWORK:-edgecloud}
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; }

echo "== 0. versions"; podman --version; echo "cgroup v2: $(test -f /sys/fs/cgroup/cgroup.controllers && echo yes || echo no)"
echo "   delegated controllers for your user: $(cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers 2>/dev/null || echo 'n/a (no systemd user session in this shell)')"

echo "== 1. image builds"
podman image exists "$IMG" && pass "image $IMG present" || { podman build -t "$IMG" -f Containerfile . && pass "built $IMG"; } || fail "build failed -> fix Containerfile/requirements first"

echo "== 2. rootless network"
podman network exists "$NET" 2>/dev/null || podman network create "$NET" >/dev/null
podman network exists "$NET" && pass "network $NET exists" || fail "cannot create network -> CNI plugin/version issue (handoff Sec. 6); check 'podman network create' error text"

echo "== 3. resource caps really apply (this is what makes 'edge' an edge)"
OUT=$(podman run --rm --cpus 1 --memory 1g "$IMG" sh -c 'echo cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null); echo memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)' 2>&1)
echo "   $OUT" | tr '\n' ' '; echo
echo "$OUT" | grep -q "cpu.max=100000 100000" && pass "cpu cap = 1 CPU" || fail "cpu cap not applied -> cpu controller not delegated to user slices. Admin (Ipek) fix: /etc/systemd/system/user@.service.d/delegate.conf with [Service] Delegate=cpu cpuset io memory pids; then systemctl daemon-reload. Rollback: delete the file."
echo "$OUT" | grep -q "memory.max=1073741824" && pass "memory cap = 1 GiB" || fail "memory cap not applied (same delegation fix)"
OUT2=$(podman run --rm --cpus 2 --memory 2g "$IMG" sh -c 'echo cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null); echo memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)' 2>&1)
echo "$OUT2" | grep -q "cpu.max=200000 100000" && echo "$OUT2" | grep -q "memory.max=2147483648" && pass "medium profile caps = 2 CPU / 2 GiB" || fail "medium profile caps not applied (same delegation fix)"

echo "== 4. netem works inside a rootless container"
OUT=$(podman run --rm --cap-add NET_ADMIN --network "$NET" "$IMG" sh -c 'tc qdisc add dev eth0 root netem delay 50ms && tc qdisc show dev eth0' 2>&1)
echo "   $OUT"
echo "$OUT" | grep -q "netem" && echo "$OUT" | grep -q "delay 50" && pass "netem delay accepted" || {
  case "$OUT" in
    *"Operation not permitted"*) fail "NET_ADMIN missing/ineffective -> check --cap-add NET_ADMIN, and that the container is on a bridge network (not slirp-only)";;
    *"No such file"*|*"unknown"*|*"Unknown qdisc"*) fail "sch_netem kernel module not loaded on host -> admin (Ipek): sudo modprobe sch_netem; persist with echo sch_netem | sudo tee /etc/modules-load.d/netem.conf. Rollback: remove that file.";;
    *"Cannot find device"*) fail "interface is not eth0 -> run: podman run --rm --network $NET $IMG ip -o link ; set NETEM_IFACE";;
    *) fail "unexpected: see output above";;
  esac; }

echo "== 5. two containers resolve each other + injected RTT is visible"
podman rm -f pf_cloud pf_edge >/dev/null 2>&1 || true
podman run -d --name pf_cloud --network "$NET" --cap-add NET_ADMIN "$IMG" sleep 120 >/dev/null
podman run -d --name pf_edge  --network "$NET" --cap-add NET_ADMIN "$IMG" sleep 120 >/dev/null
sleep 1
podman exec pf_edge  tc qdisc replace dev eth0 root netem delay 25ms 2>/dev/null
podman exec pf_cloud tc qdisc replace dev eth0 root netem delay 25ms 2>/dev/null
PING=$(podman exec pf_edge ping -c 5 -i 0.3 pf_cloud 2>&1 | tail -1)
echo "   ping -> $PING"
# expect avg ~50 ms (25 + 25)
echo "$PING" | awk -F'/' '{ if ($5+0 > 45 && $5+0 < 60) exit 0; else exit 1 }' && pass "ping avg ~50 ms: injected delay is visible in the data" || fail "ping avg not ~50 ms -> netem not in the path; check 'tc qdisc show' in both containers"
podman rm -f pf_cloud pf_edge >/dev/null 2>&1
echo "== done"
