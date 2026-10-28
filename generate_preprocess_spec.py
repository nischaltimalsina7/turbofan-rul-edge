import json
import pandas as pd

columns = (
    ["engine_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None
)

df.columns = columns

removed = [
    "op_setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_19"
]

df = df.drop(columns=removed)

features = list(df.columns[2:])

selected_features = [
    "op_setting_1", "op_setting_2",
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13",
    "sensor_14", "sensor_15",
    "sensor_17", "sensor_18",
    "sensor_20", "sensor_21"
]

spec = {
    "dataset": "CMAPSS_FD001",
    "normalization": "MinMaxScaler",
    "window_size": 30,
    "rul_cap": 130,
    "raw_column_count": len(columns),
    "selected_feature_count": len(selected_features),
    "selected_features": selected_features,
    "selected_raw_column_indices_zero_based": [
        columns.index(feature) for feature in selected_features
    ],
    "normalization_features": features,
    "train_min": {
        feature: float(df[feature].min())
        for feature in features
    },
    "train_max": {
        feature: float(df[feature].max())
        for feature in features
    }
}

with open("models/preprocess_spec.json", "w") as f:
    json.dump(spec, f, indent=2)

print("Preprocessing specification created!")
print("Raw columns:", len(columns))
print("Selected features:", len(selected_features))
print("Normalization features:", len(features))

