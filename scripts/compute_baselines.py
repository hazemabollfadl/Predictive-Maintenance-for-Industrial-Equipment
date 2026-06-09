"""
Computes per-sensor mean, min, and max from the FD001 training data
and writes lambda/baselines.json.

Run once at deploy time — output is bundled inside the Lambda package.
"""

import json
import os
import pandas as pd

SENSOR_COLUMNS = [
    "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_6", "Sensor_7",
    "Sensor_8", "Sensor_9", "Sensor_11", "Sensor_12", "Sensor_13",
    "Sensor_14", "Sensor_15", "Sensor_17", "Sensor_20", "Sensor_21",
]

COLUMNS = (
    ["Engine_ID", "Cycle", "Op_Setting_1", "Op_Setting_2", "Op_Setting_3"]
    + [f"Sensor_{i}" for i in range(1, 22)]
)

OUTPUT_PATH = "lambda/baselines.json"


def main():
    base_path = os.environ.get("DATASET_BASE_PATH", "src/data/raw")
    path = os.path.join(base_path, "training_data", "train_FD001.txt")

    df = pd.read_csv(path, sep=r"\s+", names=COLUMNS, index_col=False)

    baselines = {
        col: {
            "mean": round(df[col].mean(), 4),
            "min":  round(df[col].min(),  4),
            "max":  round(df[col].max(),  4),
        }
        for col in SENSOR_COLUMNS
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"Baselines written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
