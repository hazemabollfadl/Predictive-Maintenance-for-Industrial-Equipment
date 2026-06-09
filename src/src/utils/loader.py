import os
import pandas as pd


COLUMNS = (
    ["Engine_ID", "Cycle", "Op_Setting_1", "Op_Setting_2", "Op_Setting_3"]
    + [f"Sensor_{i}" for i in range(1, 22)]
)

ZERO_VARIANCE_COLS = [
    "Op_Setting_1", "Op_Setting_2", "Op_Setting_3",
    "Sensor_1", "Sensor_5", "Sensor_10", "Sensor_16", "Sensor_18", "Sensor_19",
]


def load_cmapss_split(base_path: str, subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load train, test, and RUL ground-truth for one C-MAPSS subset.

    Drops zero-variance sensors and sets (Engine_ID, Cycle) as a composite index.

    Returns:
        df_train, df_test, rul_series
    """
    def _read(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep=r"\s+", names=COLUMNS, index_col=False)
        df.set_index(["Engine_ID", "Cycle"], inplace=True)
        df.drop(columns=ZERO_VARIANCE_COLS, inplace=True, errors="ignore")
        return df

    df_train = _read(os.path.join(base_path, "training_data", f"train_{subset}.txt"))
    df_test = _read(os.path.join(base_path, "test_data", f"test_{subset}.txt"))
    rul = pd.read_csv(
        os.path.join(base_path, "rul_data", f"RUL_{subset}.txt"),
        header=None,
        names=["RUL"],
    ).squeeze()

    return df_train, df_test, rul
