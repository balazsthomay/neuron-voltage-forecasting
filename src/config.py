"""
Configuration module for LSTM voltage forecasting system.

This module contains all configuration parameters using dataclasses
for type safety and IDE support as specified in project standards.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch


@dataclass
class ModelConfig:
    """Configuration for LSTM model architecture."""
    
    input_size: int = 100  # Number of neurons
    hidden_size: int = 256  # Hidden units in LSTM layers
    num_layers: int = 2  # Number of LSTM layers
    output_size: int = 100  # Output neurons (same as input)
    dropout: float = 0.3  # Dropout rate
    sequence_length: int = 100  # Input sequence length
    
    def __post_init__(self) -> None:
        """Validate model configuration parameters."""
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    grad_clip_max_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-6
    
    # Learning rate scheduler
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6
    
    # Data loading
    num_workers: int = 0  # Set to 0 for MPS compatibility
    pin_memory: bool = False  # Disable for MPS
    
    def __post_init__(self) -> None:
        """Validate training configuration parameters."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")


@dataclass
class DataConfig:
    """Configuration for data handling."""
    
    data_path: str = "data/preprocessed/datasets.pt"
    normalization_params_path: str = "data/preprocessed/normalization_params.pkl"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    shuffle_data: bool = True
    random_seed: int = 42
    
    def __post_init__(self) -> None:
        """Validate data configuration parameters."""
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        if not all(0 < split < 1 for split in [self.train_split, self.val_split, self.test_split]):
            raise ValueError("All data splits must be between 0 and 1")


@dataclass
class DeviceConfig:
    """Configuration for device handling with MPS support."""
    
    device: Optional[str] = None
    force_cpu: bool = False
    
    def __post_init__(self) -> None:
        """Automatically detect and configure the best available device."""
        if self.force_cpu:
            self.device = "cpu"
        elif self.device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
    
    @property
    def torch_device(self) -> torch.device:
        """Get torch.device object."""
        return torch.device(self.device)


@dataclass
class PathConfig:
    """Configuration for model and logging paths."""
    
    models_dir: str = "models"
    best_model_path: str = "models/best_model.pth"
    latest_model_path: str = "models/latest_model.pth"
    log_file: str = "training.log"
    
    def get_checkpoint_path(self, epoch: int) -> str:
        """Generate checkpoint path for specific epoch."""
        return f"{self.models_dir}/checkpoint_epoch_{epoch}.pth"


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    device: DeviceConfig = DeviceConfig()
    paths: PathConfig = PathConfig()
    
    def __post_init__(self) -> None:
        """Initialize all sub-configurations."""
        # Ensure all dataclass post_init methods are called
        if not hasattr(self.model, '_post_init_called'):
            self.model.__post_init__()
            self.model._post_init_called = True
            
        if not hasattr(self.training, '_post_init_called'):
            self.training.__post_init__()
            self.training._post_init_called = True
            
        if not hasattr(self.data, '_post_init_called'):
            self.data.__post_init__()
            self.data._post_init_called = True
            
        if not hasattr(self.device, '_post_init_called'):
            self.device.__post_init__()
            self.device._post_init_called = True
    
    def validate_compatibility(self) -> None:
        """Validate cross-configuration compatibility."""
        # Ensure model input/output sizes match data expectations
        if self.model.input_size != self.model.output_size:
            raise ValueError("For voltage forecasting, input_size must equal output_size")
        
        # Validate device compatibility
        if self.device.device == "mps" and self.training.num_workers > 0:
            raise ValueError("num_workers must be 0 when using MPS device")