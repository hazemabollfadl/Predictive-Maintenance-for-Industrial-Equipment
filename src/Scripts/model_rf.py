import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# RANDOM FOREST FEATURE ENGINEERING


def preprocess_rf_data(df, is_train=True, rul_cap=130, dropped_sensors=None):
    """
    Applies the specific feature engineering pipeline.
    Locks dropped sensors during training to enforce symmetry during testing.
    """
    df_rf = df.copy()

    # 1. RUL Capping
    if 'RUL' in df_rf.columns and is_train:
        df_rf['RUL'] = df_rf['RUL'].clip(upper=rul_cap)

    # 2. Drop constant sensors
    sensor_cols = [c for c in df_rf.columns if c.startswith('Sensor_')]

    if is_train:
        # Dynamically calculate and record dropped sensors during training
        stats = df_rf[sensor_cols].std()
        constant_sensors = stats[stats < 0.01].index.tolist()
    else:
        # Force strict compliance to training geometry during inference
        constant_sensors = dropped_sensors if dropped_sensors is not None else []

    df_rf.drop(columns=constant_sensors, inplace=True, errors='ignore')
    useful_sensors = [s for s in sensor_cols if s not in constant_sensors]

    # 3. Lag Features
    LAG_STEPS = [1, 2, 3]
    for lag in LAG_STEPS:
        for col in useful_sensors:
            df_rf[f'{col}_lag{lag}'] = df_rf.groupby(
                level='Engine_ID')[col].shift(lag)

    # 4. Rolling Statistics
    for window in [5, 30]:
        for col in useful_sensors:
            df_rf[f'{col}_roll_mean{window}'] = df_rf.groupby(level='Engine_ID')[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df_rf[f'{col}_roll_std{window}'] = df_rf.groupby(level='Engine_ID')[col].transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )

    # Drop NaNs introduced by shifts
    df_rf.dropna(inplace=True)

    return df_rf, constant_sensors

# RANDOM FOREST TRAINING ENGINE


def train_rf_model(X_train, y_train):
    """
    Trains the baseline Random Forest based on the teammate's chosen hyperparameters.
    """
    print("--- INITIATING RANDOM FOREST TRAINING ---")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        max_features='sqrt',
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    print("Random Forest compilation and training complete.")
    return rf
