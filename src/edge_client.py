
import argparse
import csv
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.preprocessing import load_spec, preprocess_window


COLUMNS = [
    "run_id", "config", "rtt_ms", "bw", "precision",
    "edge_profile", "rep", "engine", "true_rul",
    "pred_rul", "total_ns", "edge_ns", "server_ns",
    "network_ns", "bytes_out", "bytes_in", "ts_utc"
]


def load_engines(limit):
    data = np.loadtxt("/app/data/test_FD001.txt")
    true_ruls = np.loadtxt("/app/data/RUL_FD001.txt").reshape(-1)

    engines = []
    for engine_id in sorted(np.unique(data[:, 0]).astype(int)):
        rows = data[data[:, 0] == engine_id]
        window = rows[-30:, 1:].astype(np.float32)

        engines.append((
            engine_id,
            window,
            float(true_ruls[engine_id - 1])
        ))

        if limit is not None and len(engines) >= limit:
            break

    return engines


def send_prediction(window, server_url):
    payload = json.dumps(
        {"window": window.tolist()}
    ).encode("utf-8")

    request = urllib.request.Request(
        server_url.rstrip("/") + "/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()

    result = json.loads(body.decode("utf-8"))
    return result, len(payload), len(body)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", required=True,
        choices=["cloud", "hybrid", "edge"]
    )
    parser.add_argument("--rtt-ms", type=int, required=True)
    parser.add_argument("--bw", required=True)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--edge-profile", required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.config == "edge":
        raise NotImplementedError(
            "Edge-only INT8 inference is not implemented yet."
        )

    if args.precision not in ("float32", "float16"):
        raise ValueError("Unsupported precision")

    server_url = os.environ.get(
        "SERVER_URL", "http://cloud:8000"
    )

    engines = load_engines(args.limit)
    spec = load_spec()

    results_path = Path("/app/results/runs.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Warm up the cloud model with three requests.
    warmup_window = engines[0][1]
    if args.config == "hybrid":
        warmup_window = preprocess_window(warmup_window, spec)
        if args.precision == "float16":
            warmup_window = warmup_window.astype(np.float16)

    for _ in range(3):
        send_prediction(warmup_window, server_url)

    file_exists = (
        results_path.exists() and results_path.stat().st_size > 0
    )

    with results_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)

        if not file_exists:
            writer.writeheader()

        for engine_id, raw_window, true_rul in engines:
            total_start = time.perf_counter_ns()
            edge_ns = 0

            if args.config == "hybrid":
                edge_start = time.perf_counter_ns()
                window = preprocess_window(raw_window, spec)

                if args.precision == "float16":
                    window = window.astype(np.float16)

                edge_ns = time.perf_counter_ns() - edge_start
            else:
                window = raw_window

            result, bytes_out, bytes_in = send_prediction(
                window, server_url
            )

            total_ns = time.perf_counter_ns() - total_start
            server_ns = int(result["server_ns"])
            network_ns = max(
                0, total_ns - edge_ns - server_ns
            )

            writer.writerow({
                "run_id": args.run_id,
                "config": args.config,
                "rtt_ms": args.rtt_ms,
                "bw": args.bw,
                "precision": args.precision,
                "edge_profile": args.edge_profile,
                "rep": args.rep,
                "engine": engine_id,
                "true_rul": true_rul,
                "pred_rul": float(result["pred_rul"]),
                "total_ns": total_ns,
                "edge_ns": edge_ns,
                "server_ns": server_ns,
                "network_ns": network_ns,
                "bytes_out": bytes_out,
                "bytes_in": bytes_in,
                "ts_utc": datetime.now(timezone.utc).isoformat()
            })

            file.flush()

            print(
                f"Engine {engine_id}: "
                f"true={true_rul:.1f}, "
                f"pred={float(result['pred_rul']):.2f}, "
                f"total={total_ns / 1e6:.2f} ms"
            )

    print("Results saved to", results_path)


if __name__ == "__main__":
    main()
