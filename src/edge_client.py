
import argparse
import atexit
import csv
import http.client
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

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


def send_prediction(window, connection):
    payload = json.dumps(
        {"window": window.tolist()}
    ).encode("utf-8")

    connection.request(
        "POST",
        "/predict",
        body=payload,
        headers={"Content-Type": "application/json"}
    )

    response = connection.getresponse()
    body = response.read()

    if response.status != 200:
        raise RuntimeError(
            f"Cloud HTTP {response.status}: {body.decode('utf-8')}"
        )

    result = json.loads(body.decode("utf-8"))
    return result, len(payload), len(body)



def load_int8_interpreter():
    interpreter = tf.lite.Interpreter(
        model_path="/app/models/model_int8.tflite"
    )
    interpreter.allocate_tensors()

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    if input_info["dtype"] != np.int8:
        raise ValueError("Expected INT8 model input")
    if output_info["dtype"] != np.int8:
        raise ValueError("Expected INT8 model output")

    return interpreter, input_info, output_info


def predict_int8(window, interpreter, input_info, output_info):
    x = np.asarray(window, dtype=np.float32)[np.newaxis, ...]

    scale, zero = input_info["quantization"]
    output_scale, output_zero = output_info["quantization"]

    if scale <= 0 or output_scale <= 0:
        raise ValueError("Invalid INT8 quantization parameters")

    quantized = np.clip(
        np.round(x / scale + zero), -128, 127
    ).astype(np.int8)

    interpreter.set_tensor(input_info["index"], quantized)
    interpreter.invoke()

    raw_prediction = interpreter.get_tensor(output_info["index"])
    prediction = (
        int(raw_prediction.reshape(-1)[0]) - output_zero
    ) * output_scale

    return float(prediction)

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
        if args.precision != "int8":
            raise ValueError("Edge-only requires --precision int8")
        interpreter, input_info, output_info = load_int8_interpreter()
    elif args.precision not in ("float32", "float16"):
        raise ValueError("Unsupported precision")

    server_url = os.environ.get(
        "SERVER_URL", "http://cloud:8000"
    )

    server_address = server_url.removeprefix("http://")
    if "/" in server_address:
        server_address = server_address.split("/", 1)[0]

    host, port = server_address.rsplit(":", 1)
    connection = http.client.HTTPConnection(
        host, int(port), timeout=60
    ) if args.config != "edge" else None
    if connection is not None:
        atexit.register(connection.close)

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
        if args.config == "edge":
            prepared = preprocess_window(warmup_window, spec)
            predict_int8(prepared, interpreter, input_info, output_info)
        else:
            send_prediction(warmup_window, connection)

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

            if args.config == "edge":
                edge_start = time.perf_counter_ns()
                prepared = preprocess_window(raw_window, spec)
                pred_rul = predict_int8(
                    prepared, interpreter, input_info, output_info
                )
                edge_ns = time.perf_counter_ns() - edge_start
                server_ns = 0
                network_ns = 0
                bytes_out = 0
                bytes_in = 0
            else:
                result, bytes_out, bytes_in = send_prediction(
                    window, connection
                )
                server_ns = int(result["server_ns"])
                pred_rul = float(result["pred_rul"])

            total_ns = time.perf_counter_ns() - total_start
            if args.config != "edge":
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
                "pred_rul": pred_rul,
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
                f"pred={pred_rul:.2f}, "
                f"total={total_ns / 1e6:.2f} ms"
            )

    print("Results saved to", results_path)


if __name__ == "__main__":
    main()
