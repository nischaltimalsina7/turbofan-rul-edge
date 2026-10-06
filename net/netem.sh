#!/usr/bin/env bash
# Apply one network condition to the edge<->cloud link, from the HOST, using
# `podman exec` into both running containers. No sudo: the containers were
# started with --cap-add NET_ADMIN, which in rootless Podman grants NET_ADMIN
# inside the container's own network namespace -- exactly enough to shape its
# own interface and nothing on the host.
#
# Usage: net/netem.sh <rtt_ms> <bw>          bw = unlimited | 10mbit | 1mbit
#        (applied to every edge container listed in EDGE_CONTAINERS and to cloud)
#        net/netem.sh clear
#        net/netem.sh show
#
# Delay is split symmetrically (rtt/2 egress on each side) so a request and its
# response each see one-way latency and `ping cloud` from the edge shows the
# full RTT. Bandwidth (netem `rate`) is applied on the EDGE side only, because
# the request payload is the only non-trivial transfer; the ~100-byte JSON
# response is not what the bandwidth factor is about. State this in Methods.
set -euo pipefail
EDGES=(${EDGE_CONTAINERS:-edge edge2})   # all edge-profile containers (small, medium)
CLOUD=${CLOUD_CONTAINER:-cloud}
IFACE=${NETEM_IFACE:-eth0}   # verify with: podman exec edge ip -o link

tc_in() { podman exec "$1" tc "${@:2}"; }

case "${1:-}" in
  clear)
    for c in "${EDGES[@]}" "$CLOUD"; do tc_in "$c" qdisc del dev "$IFACE" root 2>/dev/null || true; done
    echo "netem cleared" ;;
  show)
    for c in "${EDGES[@]}" "$CLOUD"; do echo "== $c"; tc_in "$c" qdisc show dev "$IFACE" 2>/dev/null || echo "(not running)"; done ;;
  *)
    RTT=${1:?rtt_ms}; BW=${2:-unlimited}
    # half-RTT with 0.1 ms resolution; tc accepts fractional ms like 12.5ms
    HALF=$(python3 -c "print(f'{$RTT/2:.1f}')")
    EDGE_RULE=(netem delay "${HALF}ms" limit 10000)
    [[ "$BW" != "unlimited" ]] && EDGE_RULE+=(rate "$BW")
    CLOUD_RULE=(netem delay "${HALF}ms" limit 10000)
    if [[ "$RTT" == "0" && "$BW" == "unlimited" ]]; then
      "$0" clear; exit 0
    fi
    for c in "${EDGES[@]}"; do tc_in "$c" qdisc replace dev "$IFACE" root "${EDGE_RULE[@]}"; done
    tc_in "$CLOUD" qdisc replace dev "$IFACE" root "${CLOUD_RULE[@]}"
    echo "netem: rtt=${RTT}ms bw=${BW} (edges ${EDGES[*]}: ${EDGE_RULE[*]} | cloud: ${CLOUD_RULE[*]})" ;;
esac
