import torch
import pytest

from src.models.cnn import CNN


@pytest.fixture
def model():
    return CNN(num_sensors=15, window_size=30).eval()


def test_output_shape(model):
    x = torch.randn(8, 30, 15)   # batch of 8 windows
    out = model(x)
    assert out.shape == (8, 1)


def test_output_is_unbounded(model):
    """Regression output must not be squeezed into [0, 1] by an activation."""
    x = torch.randn(64, 30, 15)
    out = model(x)
    # With random weights some outputs will naturally fall outside [0, 1]
    assert out.max().item() > 1.0 or out.min().item() < 0.0


def test_single_sample(model):
    x = torch.randn(1, 30, 15)
    out = model(x)
    assert out.shape == (1, 1)


def test_no_nan_in_output(model):
    x = torch.randn(16, 30, 15)
    out = model(x)
    assert not torch.isnan(out).any()


def test_custom_window_size():
    """Architecture must adapt correctly to a non-default window size."""
    model = CNN(num_sensors=15, window_size=50).eval()
    x = torch.randn(4, 50, 15)
    out = model(x)
    assert out.shape == (4, 1)
