import json
import numpy as np

SPEC_PATH = "/app/models/preprocess_spec.json"


def load_spec(path=SPEC_PATH):
    with open(path) as f:
        return json.load(f)


def preprocess_window(raw_window, spec=None):
    if spec is None:
        spec = load_spec()

    raw = np.asarray(raw_window, dtype=np.float32)

    if raw.shape != (30, 25):
        raise ValueError(
            f"Expected (30, 25), got {raw.shape}"
        )

    # The raw 25-column window excludes engine ID.
    # The saved indices refer to the original 26-column data.
    indices = [
        i - 1
        for i in spec["selected_raw_column_indices_zero_based"]
    ]

    selected = raw[:, indices].astype(np.float64)

    for j, feature in enumerate(spec["selected_features"]):
        minimum = spec["train_min"][feature]
        maximum = spec["train_max"][feature]

        if maximum > minimum:
            selected[:, j] = (
                (selected[:, j] - minimum)
                / (maximum - minimum)
            )
        else:
            selected[:, j] = 0.0

    return selected.astype(np.float32)
