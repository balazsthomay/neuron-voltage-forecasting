"""
Data preprocessing pipeline for neural activity prediction project.
This script loads raw simulation data, applies preprocessing, and creates
train/val/test datasets ready for LSTM training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, List, Dict
import pickle
from tqdm import tqdm


class NeuralVoltageDataset(Dataset):
    """PyTorch dataset for neural voltage time series."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: Input sequences of shape (n_samples, seq_len, n_neurons)
            targets: Target values of shape (n_samples, n_neurons)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def detect_spikes(voltages: np.ndarray, threshold: float = -50.0, 
                  refractory_samples: int = 20) -> np.ndarray:
    """
    Detect spikes from voltage traces using threshold crossing.
    
    Args:
        voltages: Voltage array of shape (n_neurons, n_timesteps)
        threshold: Spike threshold in mV
        refractory_samples: Minimum samples between spikes (2ms at 0.1ms dt = 20 samples)
    
    Returns:
        Binary spike array of shape (n_neurons, n_timesteps)
    """
    n_neurons, n_timesteps = voltages.shape
    spikes = np.zeros_like(voltages, dtype=bool)
    
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


def load_simulation_data(data_dir: str, n_runs: int = 50) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load all simulation runs from the output directory.
    
    Args:
        data_dir: Directory containing simulation outputs
        n_runs: Number of simulation runs to load
    
    Returns:
        Tuple of (voltage_traces_list, spike_times_list)
    """
    voltage_traces = []
    spike_times = []
    
    for run_id in tqdm(range(n_runs), desc="Loading simulation data"):
        # Load voltage data
        voltage_file = os.path.join(data_dir, f'run_{run_id}_voltages.dat')
        voltage_data = np.loadtxt(voltage_file)
        
        # Extract time column and voltage matrix
        times = voltage_data[:, 0]
        voltages = voltage_data[:, 1:].T  # Shape: (n_neurons, n_timesteps)
        
        # Load spike data for validation
        spike_file = os.path.join(data_dir, f'run_{run_id}_spikes.dat')
        spikes = np.loadtxt(spike_file)
        
        voltage_traces.append(voltages)
        spike_times.append(spikes)
    
    return voltage_traces, spike_times


def preprocess_voltages(voltage_traces: List[np.ndarray], 
                       discard_initial_ms: int = 1000,
                       add_noise: bool = True,
                       noise_std: float = 0.3) -> Tuple[np.ndarray, Dict]:
    """
    Preprocess voltage traces: discard transients, normalize, add noise.
    
    Args:
        voltage_traces: List of voltage arrays from each run
        discard_initial_ms: Milliseconds to discard from start (transient dynamics)
        add_noise: Whether to add measurement noise
        noise_std: Standard deviation of Gaussian noise in mV
    
    Returns:
        Tuple of (processed_voltages, normalization_params)
    """
    processed_runs = []
    
    # Process each run
    for voltages in voltage_traces:
        # Discard initial transient
        voltages = voltages[:, discard_initial_ms:]
        
        # Add measurement noise if requested
        if add_noise:
            noise = np.random.normal(0, noise_std, voltages.shape)
            voltages = voltages + noise
        
        processed_runs.append(voltages)
    
    # Concatenate all runs along time axis
    all_voltages = np.hstack(processed_runs)  # Shape: (n_neurons, total_timesteps)
    
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
        'discard_initial_ms': discard_initial_ms
    }
    
    return normalized_voltages, normalization_params


def create_sequences(voltages: np.ndarray, 
                    sequence_length: int = 100,
                    stride: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping sequences for time series prediction.
    
    Args:
        voltages: Normalized voltage array of shape (n_neurons, n_timesteps)
        sequence_length: Length of input sequences in timesteps
        stride: Stride between sequences (overlap = sequence_length - stride)
    
    Returns:
        Tuple of (sequences, targets) arrays
    """
    n_neurons, n_timesteps = voltages.shape
    sequences = []
    targets = []
    
    # Create sequences with sliding window
    for start_idx in range(0, n_timesteps - sequence_length - 1, stride):
        # Input sequence
        seq = voltages[:, start_idx:start_idx + sequence_length].T  # (seq_len, n_neurons)
        # Target is next timestep
        target = voltages[:, start_idx + sequence_length]  # (n_neurons,)
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def temporal_train_val_test_split(sequences: np.ndarray, 
                                 targets: np.ndarray,
                                 train_frac: float = 0.7,
                                 val_frac: float = 0.15) -> Dict:
    """
    Split data temporally to test generalization to future timepoints.
    
    Args:
        sequences: Input sequences array
        targets: Target values array
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation
    
    Returns:
        Dictionary with train/val/test datasets
    """
    n_samples = len(sequences)
    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))
    
    # Split data temporally
    train_sequences = sequences[:train_end]
    train_targets = targets[:train_end]
    
    val_sequences = sequences[train_end:val_end]
    val_targets = targets[train_end:val_end]
    
    test_sequences = sequences[val_end:]
    test_targets = targets[val_end:]
    
    # Create PyTorch datasets
    datasets = {
        'train': NeuralVoltageDataset(train_sequences, train_targets),
        'val': NeuralVoltageDataset(val_sequences, val_targets),
        'test': NeuralVoltageDataset(test_sequences, test_targets)
    }
    
    print(f"Dataset sizes - Train: {len(datasets['train'])}, "
          f"Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    
    return datasets


def save_preprocessed_data(datasets: Dict, 
                          normalization_params: Dict,
                          spike_data: List[np.ndarray],
                          output_dir: str):
    """Save preprocessed datasets and parameters for later use."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    torch.save(datasets, os.path.join(output_dir, 'datasets.pt'))
    
    # Save normalization parameters
    with open(os.path.join(output_dir, 'normalization_params.pkl'), 'wb') as f:
        pickle.dump(normalization_params, f)
    
    # Save spike data for functional evaluation
    np.save(os.path.join(output_dir, 'spike_times.npy'), spike_data)
    
    print(f"Preprocessed data saved to {output_dir}")


def main():
    """Main preprocessing pipeline."""
    # Configuration
    data_dir = 'output_raw'  # Now relative to data/ folder
    output_dir = 'preprocessed'  # Now relative to data/ folder
    n_runs = 50
    sequence_length = 100  # 100ms windows
    stride = 10  # 10ms stride for 90% overlap
    
    # Load raw data
    print("Loading simulation data...")
    voltage_traces, spike_times = load_simulation_data(data_dir, n_runs)
    
    # Preprocess voltages
    print("Preprocessing voltage traces...")
    normalized_voltages, normalization_params = preprocess_voltages(
        voltage_traces, 
        discard_initial_ms=1000,
        add_noise=True,
        noise_std=0.3
    )
    
    # Detect spikes from normalized voltages (for analysis)
    print("Detecting spikes...")
    all_spikes = []
    for i, voltages in enumerate(voltage_traces):
        # Apply same preprocessing to detect spikes
        voltages = voltages[:, 1000:]  # Discard same initial period
        spikes = detect_spikes(voltages, threshold=-50.0)
        all_spikes.append(spikes)
    
    # Create sequences
    print("Creating sequences...")
    sequences, targets = create_sequences(
        normalized_voltages, 
        sequence_length=sequence_length,
        stride=stride
    )
    
    # Split data
    print("Creating train/val/test splits...")
    datasets = temporal_train_val_test_split(sequences, targets)
    
    # Save everything
    save_preprocessed_data(datasets, normalization_params, all_spikes, output_dir)
    
    # Print summary statistics
    print("\nPreprocessing complete!")
    print(f"Total sequences created: {len(sequences)}")
    print(f"Sequence shape: {sequences[0].shape}")
    print(f"Target shape: {targets[0].shape}")
    print(f"Voltage statistics - Mean: {normalized_voltages.mean():.3f}, "
          f"Std: {normalized_voltages.std():.3f}")


if __name__ == "__main__":
    main()