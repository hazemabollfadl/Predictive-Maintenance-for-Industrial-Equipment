import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Check for hardware acceleration
device = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")

# PHASE 4: NETWORK TOPOLOGY


class CNN(nn.Module):
    def __init__(self, num_sensors):
        super(CNN, self).__init__()
        # Feature Extraction
        self.conv1 = nn.Conv1d(in_channels=num_sensors,
                               out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Flattening
        self.flatten = nn.Flatten()

        # Linear Regression Head
        self.fc1 = nn.Linear(32 * 14, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 1D convolutions strictly require the timeline to be the last dimension
        x = x.permute(0, 2, 1)

        # Extraction
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Regression
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)

        # Continuous RUL Output (Unbounded)
        x = self.fc2(x)
        return x

# PHASE 5: TRAINING ENGINE


def train_cnn(X_train, Y_train, epochs=15, batch_size=64):
    """
    Initializes, scales targets, and trains the CNN.
    Returns the trained model and the fitted target scaler for inference.
    """
    print(f"Active Device: {device}\n")
    model = CNN(num_sensors=X_train.shape[2]).to(device)

    # --- TARGET SCALING ---
    Y_train_reshaped = Y_train.reshape(-1, 1)
    target_scaler = MinMaxScaler()
    Y_train_scaled = target_scaler.fit_transform(Y_train_reshaped)

    # Convert to Tensors
    tensor_X = torch.tensor(X_train, dtype=torch.float32).to(device)
    tensor_Y = torch.tensor(Y_train_scaled, dtype=torch.float32).to(device)

    dataset = TensorDataset(tensor_X, tensor_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    print("--- INITIATING NETWORK TRAINING (SCALED TARGETS) ---")

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(
            f"Epoch {epoch+1:02d}/{epochs} | Average MSE Loss: {avg_loss:.4f}")

    return model, target_scaler


# Execute a test run only if the script is run directly
if __name__ == "__main__":
    from data_pipeline import X_train, Y_train
    print("--- TESTING CNN MODULE ---")
    trained_model, fitted_scaler = train_cnn(
        X_train, Y_train, epochs=3)  # Short test run
    print("CNN compilation and training loop successful.")
