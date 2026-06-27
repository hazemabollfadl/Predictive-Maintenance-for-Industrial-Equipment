import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import the unified data pipeline and models
from data_pipeline import df_train, X_train, Y_train, scaler, sensor_columns, cols_to_drop, subset, testing_data_path, rul_data_path, WINDOW_SIZE, columns
from model_cnn import train_cnn, device
from model_rf import preprocess_rf_data, train_rf_model

if __name__ == "__main__":
    print("==============================================")
    print(" INITIATING MASTER EVALUATION PIPELINE")
    print("==============================================\n")

    # 1. Train Both Models on the Shared Data Source
    print("[1/3] Training Models...")
    # Train CNN
    cnn_model, target_scaler = train_cnn(X_train, Y_train, epochs=15)

    # Train RF (Unpack the dropped sensors list to lock the geometry)
    df_train_rf, rf_dropped_sensors = preprocess_rf_data(
        df_train, is_train=True)
    feature_cols_rf = [
        c for c in df_train_rf.columns if c not in ['Engine_ID', 'RUL']]

    X_train_rf = df_train_rf[feature_cols_rf].values
    y_train_rf = df_train_rf['RUL'].values
    rf_model = train_rf_model(X_train_rf, y_train_rf)

    # 2. Acquire and Clean the Holdout Test Set
    print("\n[2/3] Preparing Holdout Test Data (test_FD001.txt)...")
    df_test = pd.read_csv(
        filepath_or_buffer=os.path.join(
            testing_data_path, f"test_{subset}.txt"),
        sep=r'\s+', names=columns, index_col=False
    )
    df_test.set_index(['Engine_ID', 'Cycle'], inplace=True)
    df_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Load the True Answers (RUL)
    true_rul = pd.read_csv(
        filepath_or_buffer=os.path.join(rul_data_path, f"RUL_{subset}.txt"),
        sep=r'\s+', header=None
    )[0].values

    # -----------------------------------------------------
    # 3A. CNN Inference Preparation
    # -----------------------------------------------------
    df_test_cnn = df_test.copy()
    df_test_cnn[sensor_columns] = scaler.transform(df_test_cnn[sensor_columns])

    X_test_cnn_list = []
    engine_ids_test = df_test_cnn.index.get_level_values('Engine_ID').unique()
    for engine_id in engine_ids_test:
        engine_data = df_test_cnn.xs(engine_id, level='Engine_ID')[
            sensor_columns].values
        X_test_cnn_list.append(engine_data[-WINDOW_SIZE:, :])

    tensor_X_test = torch.tensor(
        np.array(X_test_cnn_list), dtype=torch.float32).to(device)

    cnn_model.eval()
    with torch.no_grad():
        scaled_predictions = cnn_model(tensor_X_test).cpu().numpy()
        cnn_predictions = target_scaler.inverse_transform(
            scaled_predictions).flatten()

    # -----------------------------------------------------
    # 3B. Random Forest Inference Preparation
    # -----------------------------------------------------
    # Force RF to drop the exact same sensors it dropped during training
    df_test_rf, _ = preprocess_rf_data(
        df_test, is_train=False, dropped_sensors=rf_dropped_sensors)

    X_test_rf_list = []
    for engine_id in engine_ids_test:
        engine_data = df_test_rf.xs(engine_id, level='Engine_ID')
        X_test_rf_list.append(engine_data[feature_cols_rf].iloc[-1].values)

    X_test_rf = np.array(X_test_rf_list)
    rf_predictions = rf_model.predict(X_test_rf)

    # 4. Evaluation and ROI Comparison
    print("\n[3/3] HOLDOUT TEST SET PERFORMANCE")
    print("==============================================")

    cnn_mae = mean_absolute_error(true_rul, cnn_predictions)
    cnn_rmse = np.sqrt(mean_squared_error(true_rul, cnn_predictions))

    rf_mae = mean_absolute_error(true_rul, rf_predictions)
    rf_rmse = np.sqrt(mean_squared_error(true_rul, rf_predictions))

    print(f"1D CNN Model        -> MAE: {cnn_mae:.2f} | RMSE: {cnn_rmse:.2f}")
    print(f"Random Forest Model -> MAE: {rf_mae:.2f} | RMSE: {rf_rmse:.2f}")
    print("==============================================")
