Serialized model artifacts — not committed to git.

Expected files
--------------
cnn_FD001.pt        — trained CNN state dict (torch.save)
scaler_FD001.pkl    — fitted StandardScaler (joblib.dump)

cnn_FD002.pt / scaler_FD002.pkl
cnn_FD003.pt / scaler_FD003.pkl
cnn_FD004.pt / scaler_FD004.pkl

Model architecture : src/src/models/cnn.py  CNN(num_sensors=15)
Training config    : src/configs/training.yaml
Save / load example:
    torch.save(model.state_dict(), "src/models/cnn_FD001.pt")
    model.load_state_dict(torch.load("src/models/cnn_FD001.pt"))
