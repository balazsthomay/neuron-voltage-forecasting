"""
LSTM Voltage Forecasting Package

A complete PyTorch implementation for forecasting voltage traces
from neural network simulations using LSTM neural networks.
"""

from .config import Config, ModelConfig, TrainingConfig, DataConfig, DeviceConfig, PathConfig
from .data_loader import DataLoader, VoltageSequenceDataset
from .lstm_forecaster import LSTMForecaster
from .trainer import LSTMTrainer, EarlyStopping, MetricsTracker

__version__ = "1.0.0"
__author__ = "Neural Dynamics Lab"

__all__ = [
    "Config",
    "ModelConfig", 
    "TrainingConfig",
    "DataConfig",
    "DeviceConfig",
    "PathConfig",
    "DataLoader",
    "VoltageSequenceDataset",
    "LSTMForecaster",
    "LSTMTrainer",
    "EarlyStopping",
    "MetricsTracker",
]