import pandas as pd
from sklearn.preprocessing import StandardScaler


SENSOR_COLUMNS = [
    "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_6", "Sensor_7",
    "Sensor_8", "Sensor_9", "Sensor_11", "Sensor_12", "Sensor_13",
    "Sensor_14", "Sensor_15", "Sensor_17", "Sensor_20", "Sensor_21",
]


def fit_transform_sensors(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None = None,
    sensor_columns: list[str] = SENSOR_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame | None, StandardScaler]:
    """Fit StandardScaler on train sensors and transform both splits in place.

    Returns the (possibly modified) DataFrames and the fitted scaler so it can
    be persisted alongside the model artifact.
    """
    scaler = StandardScaler()
    df_train[sensor_columns] = scaler.fit_transform(df_train[sensor_columns])

    if df_test is not None:
        df_test[sensor_columns] = scaler.transform(df_test[sensor_columns])

    return df_train, df_test, scaler
