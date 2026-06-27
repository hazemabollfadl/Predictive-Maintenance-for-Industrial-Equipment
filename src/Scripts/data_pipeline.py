import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

# PHASE 1: ACQUISITION & PROFILING
load_dotenv()
base_path = os.getenv("DATASET_BASE_PATH")
print(base_path)
if base_path is None:
    raise ValueError(
        "DATASET_BASE_PATH is not set. Please check your .env file.")

trainning_data_path = os.path.join(base_path, "training_data")
testing_data_path = os.path.join(base_path, "test_data")
rul_data_path = os.path.join(base_path, "rul_data")
subset = "FD001"

columns = ['Engine_ID', 'Cycle', 'Op_Setting_1', 'Op_Setting_2', 'Op_Setting_3'] + \
          [f'Sensor_{i}' for i in range(1, 22)]

df_train = pd.read_csv(
    filepath_or_buffer=os.path.join(
        trainning_data_path, f"train_{subset}.txt"),
    sep=r'\s+',
    names=columns,
    index_col=False
)

df_train.set_index(['Engine_ID', 'Cycle'], inplace=True)

engine_lifespans = df_train.groupby(level='Engine_ID').size()

cols_to_drop = ['Op_Setting_1', 'Op_Setting_2', 'Op_Setting_3',
                'Sensor_1', 'Sensor_5', 'Sensor_10', 'Sensor_16',
                'Sensor_18', 'Sensor_19']
df_train.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# PHASE 2: TARGET ENGINEERING (RUL)
max_cycles_per_row = df_train.index.get_level_values(
    'Engine_ID').map(engine_lifespans)
df_train['RUL'] = max_cycles_per_row - df_train.index.get_level_values('Cycle')


# PHASE 3: SCALING & 3D TRANSFORMATION
sensor_columns = df_train.columns.drop(['RUL'])

scaler = StandardScaler()
df_train[sensor_columns] = scaler.fit_transform(df_train[sensor_columns])


def generate_3d_transformation(df=df_train, window_size=30):
    """
    Slides a window across each engine's timeline to extract 3D blocks.
    Returns:
        X: 3D numpy array of shape (Samples, Time_Steps, Features)
        Y: 1D numpy array of continuous RUL targets
    """
    X_list = []
    Y_list = []
    engine_ids = df.index.get_level_values('Engine_ID').unique()

    for engine_id in engine_ids:
        engine_data = df.xs(engine_id, level='Engine_ID')
        sensor_matrix = engine_data[sensor_columns].values
        label_vector = engine_data['RUL'].values
        num_rows = len(engine_data)

        for current_row in range(window_size, num_rows + 1):
            block_X = sensor_matrix[current_row - window_size: current_row, :]
            block_Y = label_vector[current_row - 1]
            X_list.append(block_X)
            Y_list.append(block_Y)

    return np.array(X_list), np.array(Y_list)


WINDOW_SIZE = 30
X_train, Y_train = generate_3d_transformation(
    df_train, window_size=WINDOW_SIZE)

# Execute only if the script is run directly (not imported)
if __name__ == "__main__":
    print("--- DATA PIPELINE VERIFICATION ---")
    print(f"Flat DataFrame (For Random Forest): {df_train.shape}")
    print(f"3D Tensor X_train (For CNN): {X_train.shape}")
    print(f"1D Tensor Y_train (For CNN): {Y_train.shape}")
    print("Pipeline execution complete. Ready for model imports.")
