"""
Data loading and preprocessing module for LSTM voltage forecasting.

This module handles loading preprocessed datasets, creating sequences,
and preparing data loaders with proper device placement and normalization.
"""

import pickle
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import Config, DataConfig, DeviceConfig


logger = logging.getLogger(__name__)


class VoltageSequenceDataset(data.Dataset):
    """PyTorch Dataset for voltage sequence forecasting."""
    
    def __init__(
        self,
        sequences: torch.Tensor,
        targets: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> None:
        """
        Initialize voltage sequence dataset.
        
        Args:
            sequences: Input sequences of shape (N, sequence_length, num_neurons)
            targets: Target values of shape (N, num_neurons)
            device: Device to place tensors on
            dtype: Data type for tensors
        """
        self.sequences = sequences.to(device=device, dtype=dtype)
        self.targets = targets.to(device=device, dtype=dtype)
        
        # Validate shapes
        if len(self.sequences) != len(self.targets):
            raise ValueError(f"Sequences and targets must have same length, got {len(sequences)} vs {len(targets)}")
        
        logger.info(f"Dataset initialized with {len(self)} samples")
        logger.info(f"Sequence shape: {self.sequences.shape}")
        logger.info(f"Target shape: {self.targets.shape}")
        logger.info(f"Device: {device}, dtype: {dtype}")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        return self.sequences[idx], self.targets[idx]
    
    def get_shape_info(self) -> Dict[str, Any]:
        """Get dataset shape information for validation."""
        return {
            'num_samples': len(self),
            'sequence_length': self.sequences.shape[1],
            'num_neurons': self.sequences.shape[2],
            'target_size': self.targets.shape[1],
            'device': self.sequences.device,
            'dtype': self.sequences.dtype
        }


class DataLoader:
    """Main data loading class with comprehensive data handling."""
    
    def __init__(self, config: Config) -> None:
        """
        Initialize data loader with configuration.
        
        Args:
            config: Main configuration object
        """
        self.config = config
        self.data_config = config.data
        self.device_config = config.device
        self.model_config = config.model
        
        self.scaler: Optional[StandardScaler] = None
        self.normalization_params: Optional[Dict[str, Any]] = None
        
        logger.info("DataLoader initialized")
    
    def load_preprocessed_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load preprocessed voltage data from file.
        
        Returns:
            Tuple of (sequences, targets) tensors
        """
        data_path = Path(self.data_config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Loading data from {data_path}")
        
        try:
            # Load data without class dependencies
            import pickle
            with open(data_path, 'rb') as f:
                # Custom unpickler to handle missing class
                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if name == 'NeuralVoltageDataset':
                            # Return a dummy class that we can extract data from
                            return object
                        return super().find_class(module, name)
                
                unpickler = CustomUnpickler(f)
                data = unpickler.load()
        except:
            # If custom unpickler fails, try loading with torch.load
            try:
                import sys
                # Add dummy class to avoid deserialization error
                sys.modules['__main__'].NeuralVoltageDataset = type('NeuralVoltageDataset', (), {
                    '__init__': lambda self, seq, tar: setattr(self, 'sequences', seq) or setattr(self, 'targets', tar)
                })
                data = torch.load(data_path, map_location='cpu')
            except Exception as e:
                raise RuntimeError(f"Could not load data file {data_path}: {e}")
        
        # Extract sequences and targets from the loaded datasets
        all_sequences = []
        all_targets = []
        
        if isinstance(data, dict):
            # Process train/val/test datasets
            for split in ['train', 'val', 'test']:
                if split in data:
                    dataset = data[split]
                    if hasattr(dataset, 'sequences') and hasattr(dataset, 'targets'):
                        all_sequences.append(dataset.sequences)
                        all_targets.append(dataset.targets)
            
            if not all_sequences:
                raise ValueError("No valid datasets found in loaded data")
            
            # Concatenate all splits
            sequences = torch.cat(all_sequences, dim=0)
            targets = torch.cat(all_targets, dim=0)
        else:
            raise ValueError("Expected dictionary with train/val/test splits")
        
        logger.info(f"Loaded sequences shape: {sequences.shape}")
        logger.info(f"Loaded targets shape: {targets.shape}")
        
        return sequences, targets
    
    def load_normalization_params(self) -> None:
        """Load normalization parameters from file."""
        params_path = Path(self.data_config.normalization_params_path)
        if params_path.exists():
            logger.info(f"Loading normalization parameters from {params_path}")
            with open(params_path, 'rb') as f:
                self.normalization_params = pickle.load(f)
            logger.info("Normalization parameters loaded successfully")
        else:
            logger.warning(f"Normalization parameters not found at {params_path}")
    
    def validate_sequences(self, sequences: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate and potentially adjust pre-created sequences and targets.
        
        Args:
            sequences: Pre-created sequences tensor
            targets: Pre-created targets tensor
            
        Returns:
            Tuple of validated (sequences, targets)
        """
        # Validate shapes
        if len(sequences.shape) != 3:
            raise ValueError(f"Expected 3D sequences (batch, seq_len, features), got shape {sequences.shape}")
        
        if len(targets.shape) != 2:
            raise ValueError(f"Expected 2D targets (batch, features), got shape {targets.shape}")
        
        batch_size, seq_len, num_features = sequences.shape
        target_batch, target_features = targets.shape
        
        if batch_size != target_batch:
            raise ValueError(f"Sequences and targets have different batch sizes: {batch_size} vs {target_batch}")
        
        if num_features != target_features:
            raise ValueError(f"Sequences and targets have different feature counts: {num_features} vs {target_features}")
        
        # Check if sequence length matches config
        expected_seq_len = self.model_config.sequence_length
        if seq_len != expected_seq_len:
            logger.warning(f"Sequence length mismatch: data has {seq_len}, config expects {expected_seq_len}")
            # Update config to match data
            self.model_config.sequence_length = seq_len
            logger.info(f"Updated model config sequence_length to {seq_len}")
        
        # Check if feature count matches config
        expected_features = self.model_config.input_size
        if num_features != expected_features:
            logger.warning(f"Feature count mismatch: data has {num_features}, config expects {expected_features}")
            # Update config to match data
            self.model_config.input_size = num_features
            self.model_config.output_size = num_features
            logger.info(f"Updated model config input/output size to {num_features}")
        
        logger.info(f"Validated {batch_size} sequences of length {seq_len} with {num_features} features")
        return sequences, targets
    
    def split_data(
        self,
        sequences: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            sequences: Input sequences
            targets: Target values
            
        Returns:
            Tuple of ((train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets))
        """
        total_samples = len(sequences)
        
        # Calculate split indices
        train_size = int(self.data_config.train_split * total_samples)
        val_size = int(self.data_config.val_split * total_samples)
        
        # Handle shuffle
        if self.data_config.shuffle_data:
            torch.manual_seed(self.data_config.random_seed)
            indices = torch.randperm(total_samples)
            sequences = sequences[indices]
            targets = targets[indices]
        
        # Split data
        train_seq = sequences[:train_size]
        train_targets = targets[:train_size]
        
        val_seq = sequences[train_size:train_size + val_size]
        val_targets = targets[train_size:train_size + val_size]
        
        test_seq = sequences[train_size + val_size:]
        test_targets = targets[train_size + val_size:]
        
        logger.info(f"Data split - Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")
        
        return (train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets)
    
    def create_data_loaders(self) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
        """
        Create complete data loading pipeline.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load preprocessed sequences and targets
        sequences, targets = self.load_preprocessed_data()
        self.load_normalization_params()
        
        # Validate sequences and adjust config if needed
        sequences, targets = self.validate_sequences(sequences, targets)
        
        # Split data
        (train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets) = self.split_data(
            sequences, targets
        )
        
        # Create datasets with device placement
        device = self.device_config.torch_device
        
        train_dataset = VoltageSequenceDataset(train_seq, train_targets, device)
        val_dataset = VoltageSequenceDataset(val_seq, val_targets, device)
        test_dataset = VoltageSequenceDataset(test_seq, test_targets, device)
        
        # Log shape validation
        train_info = train_dataset.get_shape_info()
        logger.info(f"Training dataset info: {train_info}")
        
        # Create data loaders with MPS-compatible settings
        loader_kwargs = {
            'batch_size': self.config.training.batch_size,
            'num_workers': self.config.training.num_workers,
            'pin_memory': self.config.training.pin_memory,
            'drop_last': False
        }
        
        train_loader = data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = data.DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        
        logger.info("Data loaders created successfully")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def get_data_stats(self, data_loader: data.DataLoader) -> Dict[str, float]:
        """
        Calculate statistics for a data loader.
        
        Args:
            data_loader: PyTorch DataLoader
            
        Returns:
            Dictionary with data statistics
        """
        all_sequences = []
        all_targets = []
        
        for sequences, targets in data_loader:
            all_sequences.append(sequences)
            all_targets.append(targets)
        
        all_sequences = torch.cat(all_sequences, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        return {
            'seq_mean': float(all_sequences.mean()),
            'seq_std': float(all_sequences.std()),
            'seq_min': float(all_sequences.min()),
            'seq_max': float(all_sequences.max()),
            'target_mean': float(all_targets.mean()),
            'target_std': float(all_targets.std()),
            'target_min': float(all_targets.min()),
            'target_max': float(all_targets.max()),
        }