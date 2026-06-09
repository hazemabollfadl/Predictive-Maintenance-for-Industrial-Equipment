import torch
import torch.nn as nn


class CNN(nn.Module):
    """1-D CNN regressor that predicts Remaining Useful Life (RUL).

    Input:  (batch, window_size, n_sensors)
    Output: (batch, 1) — raw RUL estimate in cycles (unbounded positive number)

    window_size must match the value used during feature engineering.
    The fc1 input size is derived automatically from window_size so the
    architecture stays valid for any window length.
    """

    def __init__(self, num_sensors: int = 15, window_size: int = 30):
        super().__init__()

        # Conv1d expects (batch, channels, length); permute is applied in forward
        self.conv1 = nn.Conv1d(in_channels=num_sensors, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # length after Conv1d (kernel=3, padding=0): window_size - 2
        # length after MaxPool1d (kernel=2):         (window_size - 2) // 2
        fc1_input_size = 32 * ((window_size - 2) // 2)
        self.fc1 = nn.Linear(fc1_input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)   # (batch, sensors, time)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.fc1(self.flatten(x)))
        return self.fc2(x)        # raw regression output — no activation
