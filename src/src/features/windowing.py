import numpy as np
import pandas as pd


SENSOR_COLUMNS = [
    "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_6", "Sensor_7",
    "Sensor_8", "Sensor_9", "Sensor_11", "Sensor_12", "Sensor_13",
    "Sensor_14", "Sensor_15", "Sensor_17", "Sensor_20", "Sensor_21",
]


def generate_3d_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    sensor_columns: list[str] = SENSOR_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a fixed-length window over each engine's timeline.

    Returns:
        X: (n_samples, window_size, n_sensors)
        y: (n_samples,) — continuous RUL value at the last cycle of each window
    """
    X_list, y_list = [], []

    engine_ids = df.index.get_level_values("Engine_ID").unique()

    for engine_id in engine_ids:
        engine_data = df.xs(engine_id, level="Engine_ID")
        sensor_matrix = engine_data[sensor_columns].values
        rul_vector = engine_data["RUL"].values
        num_rows = len(engine_data)

        for end in range(window_size, num_rows + 1):
            X_list.append(sensor_matrix[end - window_size : end, :])
            y_list.append(rul_vector[end - 1])

    return np.array(X_list), np.array(y_list, dtype=np.float32)
