import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_window, load_spec

SPEC = load_spec()
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/best_model.keras")
MODEL = load_model(MODEL_PATH, compile=False)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/predict":
            self.send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1000000:
                self.send_json(400, {"error": "invalid request size"})
                return

            request = json.loads(self.rfile.read(length))
            start = time.perf_counter_ns()
            data = np.asarray(request["window"], dtype=np.float32)

            if data.shape == (30, 25):
               data = preprocess_window(data, SPEC)
            elif data.shape != (30, 17):
               self.send_json(
                   400,
                  {"error": "expected window shape (30, 25) or (30, 17)"}
                )
               return
            prediction = MODEL(data[np.newaxis, :, :], training=False)
            predicted_rul = float(prediction.numpy()[0, 0])
            server_ns = time.perf_counter_ns() - start

            self.send_json(200, {
                "pred_rul": predicted_rul,
                "server_ns": server_ns
            })

        except (ValueError, TypeError, KeyError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            print("Prediction error:", repr(exc), flush=True)
            self.send_json(500, {"error": "prediction failed"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Cloud server listening on port {port}", flush=True)
    server.serve_forever()
