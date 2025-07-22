"""
Integrated data processing and loading module for LSTM voltage forecasting.

This module handles the complete data pipeline from raw simulation files
to training-ready DataLoaders, including preprocessing, sequence creation,
and device-optimized batch loading.
"""

import logging
import os
import pickle
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

from .config import Config


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


class DataProcessor:
    """Handles raw data processing and preprocessing operations."""
    
    def __init__(self, config: Config) -> None:
        """
        Initialize data processor.
        
        Args:
            config: Main configuration object
        """
        self.config = config
        self.data_config = config.data
        logger.info("DataProcessor initialized")
    
    def detect_spikes(self, voltages: np.ndarray) -> np.ndarray:
        """
        Detect spikes from voltage traces using threshold crossing.
        
        Args:
            voltages: Voltage array of shape (n_neurons, n_timesteps)
            
        Returns:
            Binary spike array of shape (n_neurons, n_timesteps)
        """
        n_neurons, n_timesteps = voltages.shape
        spikes = np.zeros_like(voltages, dtype=bool)
        
        threshold = self.data_config.spike_threshold
        refractory_samples = self.data_config.refractory_samples
        
        for neuron_idx in range(n_neurons):
            v = voltages[neuron_idx]
            # Find threshold crossings
            crossings = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0] + 1
            
            # Enforce refractory period
            if len(crossings) > 0:
                valid_spikes = [crossings[0]]
                for spike_time in crossings[1:]:
                    if spike_time - valid_spikes[-1] >= refractory_samples:
                        valid_spikes.append(spike_time)
                
                spikes[neuron_idx, valid_spikes] = True
        
        return spikes
    
    def load_simulation_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load all simulation runs from the raw data directory.
        
        Returns:
            Tuple of (voltage_traces_list, spike_times_list)
        """
        voltage_traces = []
        spike_times = []
        
        data_dir = self.data_config.raw_data_dir
        n_runs = self.data_config.n_simulation_runs
        
        logger.info(f"Loading {n_runs} simulation runs from {data_dir}")
        
        for run_id in tqdm(range(n_runs), desc="Loading simulation data"):
            # Load voltage data
            voltage_file = os.path.join(data_dir, f'run_{run_id}_voltages.dat')
            if not os.path.exists(voltage_file):
                logger.warning(f"Voltage file not found: {voltage_file}")
                continue
                
            voltage_data = np.loadtxt(voltage_file)
            
            # Extract time column and voltage matrix
            times = voltage_data[:, 0]
            voltages = voltage_data[:, 1:].T  # Shape: (n_neurons, n_timesteps)
            
            # Load spike data for validation (optional)
            spike_file = os.path.join(data_dir, f'run_{run_id}_spikes.dat')
            if os.path.exists(spike_file):
                spikes = np.loadtxt(spike_file)
                spike_times.append(spikes)
            else:
                spike_times.append(np.array([]))  # Empty array if no spike data
            
            voltage_traces.append(voltages)
        
        logger.info(f"Loaded {len(voltage_traces)} voltage traces")
        return voltage_traces, spike_times
    
    def preprocess_voltages(self, voltage_traces: List[np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess voltage traces: discard transients, normalize, add noise.
        
        Args:
            voltage_traces: List of voltage arrays from each run
            
        Returns:
            Tuple of (processed_voltages, normalization_params)
        """
        processed_runs = []
        discard_ms = self.data_config.discard_initial_ms
        
        logger.info(f"Preprocessing {len(voltage_traces)} voltage traces")
        logger.info(f"Discarding initial {discard_ms}ms, adding noise: {self.data_config.add_noise}")
        
        # Process each run
        for voltages in voltage_traces:
            # Discard initial transient
            voltages = voltages[:, discard_ms:]
            
            # Add measurement noise if requested
            if self.data_config.add_noise:
                noise = np.random.normal(0, self.data_config.noise_std, voltages.shape)
                voltages = voltages + noise
            
            processed_runs.append(voltages)
        
        # Concatenate all runs along time axis
        all_voltages = np.hstack(processed_runs)  # Shape: (n_neurons, total_timesteps)
        
        logger.info(f"Combined voltage shape: {all_voltages.shape}")
        
        # Calculate normalization parameters (per-neuron z-score)
        means = np.mean(all_voltages, axis=1, keepdims=True)
        stds = np.std(all_voltages, axis=1, keepdims=True)
        
        # Avoid division by zero
        stds[stds == 0] = 1.0
        
        # Normalize
        normalized_voltages = (all_voltages - means) / stds
        
        normalization_params = {
            'means': means,
            'stds': stds,
            'discard_initial_ms': discard_ms,
            'noise_std': self.data_config.noise_std if self.data_config.add_noise else 0.0
        }
        
        logger.info(f"Normalization - Mean: {normalized_voltages.mean():.6f}, Std: {normalized_voltages.std():.6f}")
        
        return normalized_voltages, normalization_params
    
    def create_sequences(self, voltages: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping sequences for time series prediction.
        
        Args:
            voltages: Normalized voltage array of shape (n_neurons, n_timesteps)
            
        Returns:
            Tuple of (sequences, targets) arrays
        """
        sequence_length = self.data_config.sequence_length
        stride = self.data_config.sequence_stride
        
        n_neurons, n_timesteps = voltages.shape
        sequences = []
        targets = []
        
        logger.info(f"Creating sequences: length={sequence_length}, stride={stride}")
        
        # Create sequences with sliding window
        for start_idx in range(0, n_timesteps - sequence_length - 1, stride):
            # Input sequence
            seq = voltages[:, start_idx:start_idx + sequence_length].T  # (seq_len, n_neurons)
            # Target is next timestep
            target = voltages[:, start_idx + sequence_length]  # (n_neurons,)
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Created {len(sequences)} sequences of shape {sequences[0].shape}")
        
        return sequences, targets
    
    def get_config_hash(self) -> str:
        """Generate hash of preprocessing configuration for cache validation."""
        config_dict = {
            'n_runs': self.data_config.n_simulation_runs,
            'discard_ms': self.data_config.discard_initial_ms,
            'add_noise': self.data_config.add_noise,
            'noise_std': self.data_config.noise_std,
            'sequence_length': self.data_config.sequence_length,
            'stride': self.data_config.sequence_stride,
        }
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def save_processed_data(self, sequences: np.ndarray, targets: np.ndarray, 
                           normalization_params: Dict) -> None:
        """Save processed data to cache."""
        cache_dir = Path(self.data_config.preprocessed_dir)
        cache_dir.mkdir(exist_ok=True)
        
        cache_data = {
            'sequences': torch.tensor(sequences, dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.float32),
            'config_hash': self.get_config_hash(),
            'normalization_params': normalization_params
        }
        
        torch.save(cache_data, self.data_config.cache_file)
        
        # Also save normalization params separately for compatibility
        with open(self.data_config.normalization_params_path, 'wb') as f:
            pickle.dump(normalization_params, f)
        
        logger.info(f"Processed data cached to {self.data_config.cache_file}")
    
    def load_cached_data(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """Load cached processed data if available and valid."""
        if not self.data_config.use_cache or self.data_config.force_reprocess:
            return None
        
        cache_file = Path(self.data_config.cache_file)
        if not cache_file.exists():
            logger.info("No cached data found")
            return None
        
        try:
            cache_data = torch.load(cache_file, map_location='cpu')
            
            # Validate config hash
            if cache_data.get('config_hash') != self.get_config_hash():
                logger.info("Cache invalidated: configuration changed")
                return None
            
            logger.info("Loading data from cache")
            return (
                cache_data['sequences'],
                cache_data['targets'], 
                cache_data['normalization_params']
            )
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def process_data(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Main data processing pipeline.
        
        Returns:
            Tuple of (sequences, targets, normalization_params)
        """
        # Try loading from cache first
        cached_data = self.load_cached_data()
        if cached_data is not None:
            return cached_data
        
        logger.info("Processing data from raw simulation files")
        
        # Load raw simulation data
        voltage_traces, spike_times = self.load_simulation_data()
        
        # Preprocess voltages
        normalized_voltages, normalization_params = self.preprocess_voltages(voltage_traces)
        
        # Create sequences
        sequences, targets = self.create_sequences(normalized_voltages)
        
        # Convert to tensors
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        # Save to cache
        self.save_processed_data(sequences, targets, normalization_params)
        
        return sequences_tensor, targets_tensor, normalization_params


class DataLoader:
    """Main data loading class with integrated preprocessing."""
    
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
        
        self.processor = DataProcessor(config)
        self.normalization_params: Optional[Dict[str, Any]] = None
        
        logger.info("DataLoader initialized")
    
    def validate_and_update_config(self, sequences: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate sequences and update model config if needed.
        
        Args:
            sequences: Input sequences tensor
            targets: Target values tensor
            
        Returns:
            Validated (sequences, targets)
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
        
        # Update model config to match data
        if seq_len != self.model_config.sequence_length:
            logger.info(f"Updating model sequence_length: {self.model_config.sequence_length} -> {seq_len}")
            self.model_config.sequence_length = seq_len
        
        if num_features != self.model_config.input_size:
            logger.info(f"Updating model input/output size: {self.model_config.input_size} -> {num_features}")
            self.model_config.input_size = num_features
            self.model_config.output_size = num_features
        
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
        Create complete data loading pipeline from raw data to DataLoaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Process data from raw files or cache
        sequences, targets, normalization_params = self.processor.process_data()
        self.normalization_params = normalization_params
        
        # Validate and update config
        sequences, targets = self.validate_and_update_config(sequences, targets)
        
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
            all_sequences.append(sequences.cpu())
            all_targets.append(targets.cpu())
        
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