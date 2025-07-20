"""
Utility functions for data validation and visualization.
Helps verify preprocessing pipeline and data quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Tuple, List
import os

# Import the missing class definition from the preprocessing script
from preprocess_data import NeuralVoltageDataset


def validate_preprocessed_data(data_dir: str = 'preprocessed'):
    """
    Load and validate preprocessed datasets, print statistics.
    
    Args:
        data_dir: Directory containing preprocessed data
    """
    print("Loading preprocessed data...")
    
    # Load datasets
    # This line will now work because NeuralVoltageDataset is defined.
    datasets = torch.load(os.path.join(data_dir, 'datasets.pt'))
    
    # Load normalization parameters
    import pickle
    with open(os.path.join(data_dir, 'normalization_params.pkl'), 'rb') as f:
        norm_params = pickle.load(f)
    
    # Print statistics for each dataset
    for split_name, dataset in datasets.items():
        print(f"\n{split_name.upper()} Dataset:")
        print(f"  Number of samples: {len(dataset)}")
        
        # Get a sample
        sample_seq, sample_target = dataset[0]
        print(f"  Sequence shape: {sample_seq.shape}")
        print(f"  Target shape: {sample_target.shape}")
        
        # Calculate statistics across all samples
        all_sequences = torch.stack([dataset[i][0] for i in range(len(dataset))])
        all_targets = torch.stack([dataset[i][1] for i in range(len(dataset))])
        
        print(f"  Sequence stats - Mean: {all_sequences.mean():.3f}, Std: {all_sequences.std():.3f}")
        print(f"  Target stats - Mean: {all_targets.mean():.3f}, Std: {all_targets.std():.3f}")
        print(f"  Min value: {all_sequences.min():.3f}, Max value: {all_sequences.max():.3f}")
    
    print(f"\nNormalization parameters:")
    print(f"  Discarded initial: {norm_params['discard_initial_ms']} ms")
    print(f"  Per-neuron means shape: {norm_params['means'].shape}")
    print(f"  Per-neuron stds shape: {norm_params['stds'].shape}")
    
    
def visualize_raw_and_processed_sequences(raw_data_dir: str = 'output_raw',
                                         preprocessed_dir: str = 'preprocessed', 
                                         n_samples: int = 3,
                                         n_neurons_plot: int = 5,
                                         sequence_length: int = 100):
    """
    Visualize both raw and processed sequences side by side.
    
    Args:
        raw_data_dir: Directory containing raw simulation data
        preprocessed_dir: Directory containing preprocessed data
        n_samples: Number of sample sequences to plot
        n_neurons_plot: Number of neurons to plot per sample
        sequence_length: Length of sequences to show
    """
    # Load preprocessed dataset
    datasets = torch.load(os.path.join(preprocessed_dir, 'datasets.pt'))
    train_dataset = datasets['train']
    
    # Load normalization parameters
    import pickle
    with open(os.path.join(preprocessed_dir, 'normalization_params.pkl'), 'rb') as f:
        norm_params = pickle.load(f)
    
    # Create figure with 3 columns: raw voltage, processed voltage, target distribution
    fig, axes = plt.subplots(n_samples, 3, figsize=(18, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # For each sample
    for i in range(n_samples):
        # Get a processed sequence
        sequence, target = train_dataset[i * 1000]  # Sample every 1000th to get variety
        
        # Determine which run and time window this sequence came from
        # (This is approximate - in practice you might want to track this during preprocessing)
        total_samples_per_run = 9000 - sequence_length  # 10000 - 1000 (discarded) - sequence_length
        run_id = (i * 1000) // total_samples_per_run
        time_offset = (i * 1000) % total_samples_per_run + 1000  # Add back discarded time
        
        # Load corresponding raw data
        voltage_file = os.path.join(raw_data_dir, f'run_{run_id}_voltages.dat')
        voltage_data = np.loadtxt(voltage_file)
        times = voltage_data[:, 0]
        raw_voltages = voltage_data[:, 1:].T  # (n_neurons, n_timesteps)
        
        # Plot raw voltage sequence
        ax = axes[i, 0]
        time_indices = range(time_offset, time_offset + sequence_length)
        for neuron_idx in range(min(n_neurons_plot, raw_voltages.shape[0])):
            ax.plot(times[time_indices], 
                   raw_voltages[neuron_idx, time_indices], 
                   alpha=0.7, label=f'Neuron {neuron_idx}')
        ax.axhline(-50, color='red', linestyle='--', alpha=0.3, label='Threshold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title(f'Sample {i}: Raw voltage sequence')
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-75, -45)
        
        # Plot processed sequence
        ax = axes[i, 1]
        for neuron_idx in range(min(n_neurons_plot, sequence.shape[1])):
            ax.plot(sequence[:, neuron_idx], alpha=0.7, label=f'Neuron {neuron_idx}')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Time step (ms)')
        ax.set_ylabel('Normalized voltage')
        ax.set_title(f'Sample {i}: Processed sequence (z-scored)')
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 3)
        
        # Plot target distribution
        ax = axes[i, 2]
        ax.hist(target.numpy(), bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(target.mean(), color='red', linestyle='--', 
                  label=f'Mean: {target.mean():.2f}')
        ax.set_xlabel('Normalized voltage')
        ax.set_ylabel('Count')
        ax.set_title(f'Sample {i}: Target (next timestep)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(preprocessed_dir, 'raw_and_processed_sequences.png'), dpi=150)
    plt.show()


def visualize_voltage_distributions(raw_data_dir: str = 'output_raw',
                                  preprocessed_dir: str = 'preprocessed',
                                  n_runs: int = 5):
    """
    Compare voltage distributions before and after preprocessing.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Collect raw voltages
    all_raw_voltages = []
    for run_id in range(min(n_runs, 5)):  # Sample first 5 runs
        voltage_file = os.path.join(raw_data_dir, f'run_{run_id}_voltages.dat')
        voltage_data = np.loadtxt(voltage_file)
        raw_voltages = voltage_data[:, 1:].T  # (n_neurons, n_timesteps)
        # Skip first 1000ms as in preprocessing
        all_raw_voltages.extend(raw_voltages[:, 1000:].flatten())
    
    # Load preprocessed data
    datasets = torch.load(os.path.join(preprocessed_dir, 'datasets.pt'))
    train_data = datasets['train']
    
    # Collect processed voltages
    all_processed_voltages = []
    for i in range(min(1000, len(train_data))):  # Sample first 1000 sequences
        seq, _ = train_data[i]
        all_processed_voltages.extend(seq.numpy().flatten())
    
    # Plot raw voltage distribution
    ax1.hist(all_raw_voltages, bins=100, alpha=0.7, color='blue', density=True)
    ax1.axvline(np.mean(all_raw_voltages), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_raw_voltages):.1f} mV')
    ax1.set_xlabel('Voltage (mV)')
    ax1.set_ylabel('Density')
    ax1.set_title('Raw Voltage Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot processed voltage distribution
    ax2.hist(all_processed_voltages, bins=100, alpha=0.7, color='green', density=True)
    ax2.axvline(0, color='red', linestyle='--', label='Mean: 0 (normalized)')
    ax2.set_xlabel('Normalized voltage')
    ax2.set_ylabel('Density')
    ax2.set_title('Processed Voltage Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot voltage traces over time (raw)
    voltage_data = np.loadtxt(os.path.join(raw_data_dir, 'run_0_voltages.dat'))
    times = voltage_data[:, 0]
    sample_neurons = [0, 10, 20, 30, 40]  # Sample neurons
    
    for idx in sample_neurons:
        ax3.plot(times[2000:2500], voltage_data[2000:2500, idx+1], 
                alpha=0.7, linewidth=0.8)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Voltage (mV)')
    ax3.set_title('Raw Voltage Traces (500ms window)')
    ax3.grid(True, alpha=0.3)
    
    # Plot processed traces
    sequences = []
    for i in range(5):  # Get 5 sequences
        seq, _ = train_data[i * 100]
        sequences.append(seq[:, sample_neurons[i]])
    
    for i, seq in enumerate(sequences):
        ax4.plot(seq.numpy(), alpha=0.7, linewidth=0.8, label=f'Neuron {sample_neurons[i]}')
    ax4.set_xlabel('Time step (ms)')
    ax4.set_ylabel('Normalized voltage')
    ax4.set_title('Processed Voltage Sequences (100ms windows)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(preprocessed_dir, 'voltage_distributions.png'), dpi=150)
    plt.show()
    
    # Print statistics
    print("\nVoltage Statistics:")
    print(f"Raw voltages: Mean={np.mean(all_raw_voltages):.2f} mV, Std={np.std(all_raw_voltages):.2f} mV")
    print(f"Processed: Mean={np.mean(all_processed_voltages):.3f}, Std={np.std(all_processed_voltages):.3f}")


def analyze_spike_statistics(raw_data_dir: str = 'output_raw', 
                             preprocessed_dir: str = 'preprocessed',
                           n_runs: int = 50):
    """
    Analyze spike statistics from raw data to verify realistic dynamics.
    
    Args:
        raw_data_dir: Directory containing raw simulation data
        n_runs: Number of runs to analyze
    """
    all_firing_rates = []
    all_cv_isis = []
    
    for run_id in range(n_runs):
        # Load spike data
        spike_file = os.path.join(raw_data_dir, f'run_{run_id}_spikes.dat')
        if not os.path.exists(spike_file):
            continue
            
        spikes = np.loadtxt(spike_file)
        if len(spikes) == 0:
            continue
            
        spike_times = spikes[:, 0]
        neuron_ids = spikes[:, 1].astype(int)
        
        # Calculate firing rates
        n_neurons = 100
        sim_duration = 10000  # 10 seconds in ms
        
        for neuron_id in range(n_neurons):
            neuron_spikes = spike_times[neuron_ids == neuron_id]
            firing_rate = len(neuron_spikes) / (sim_duration / 1000)  # Hz
            all_firing_rates.append(firing_rate)
            
            # Calculate CV of ISI
            if len(neuron_spikes) > 2:
                isis = np.diff(neuron_spikes)
                cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
                all_cv_isis.append(cv)
    
    # Plot statistics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Firing rate distribution
    ax1.hist(all_firing_rates, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(np.mean(all_firing_rates), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_firing_rates):.1f} Hz')
    ax1.set_xlabel('Firing rate (Hz)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Firing Rates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CV ISI distribution
    ax2.hist(all_cv_isis, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(np.mean(all_cv_isis), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_cv_isis):.2f}')
    ax2.axvline(1.0, color='blue', linestyle=':', label='CV=1 (Poisson)')
    ax2.set_xlabel('CV of ISI')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of ISI Variability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(preprocessed_dir, 'spike_statistics.png'), dpi=150)
    plt.show()
    
    print(f"\nSpike Statistics Summary:")
    print(f"Mean firing rate: {np.mean(all_firing_rates):.2f} ± {np.std(all_firing_rates):.2f} Hz")
    print(f"Mean CV ISI: {np.mean(all_cv_isis):.3f} ± {np.std(all_cv_isis):.3f}")
    print(f"Fraction of neurons with CV > 0.8: {np.mean(np.array(all_cv_isis) > 0.8):.2%}")


def compare_raw_and_processed(raw_data_dir: str = 'output_raw',
                            preprocessed_dir: str = 'preprocessed',
                            run_id: int = 0,
                            neuron_id: int = 0,
                            time_window: Tuple[int, int] = (2000, 3000)):
    """
    Compare raw and processed voltage traces for validation.
    
    Args:
        raw_data_dir: Directory with raw data
        preprocessed_dir: Directory with preprocessed data
        run_id: Which run to examine
        neuron_id: Which neuron to plot
        time_window: Time window to plot (start_ms, end_ms)
    """
    # Load raw data
    voltage_file = os.path.join(raw_data_dir, f'run_{run_id}_voltages.dat')
    voltage_data = np.loadtxt(voltage_file)
    times = voltage_data[:, 0]
    raw_voltages = voltage_data[:, 1:].T  # (n_neurons, n_timesteps)
    
    # Load normalization parameters
    import pickle
    with open(os.path.join(preprocessed_dir, 'normalization_params.pkl'), 'rb') as f:
        norm_params = pickle.load(f)
    
    # Apply same preprocessing
    discard_ms = norm_params['discard_initial_ms']
    processed_voltage = raw_voltages[neuron_id, discard_ms:]
    
    # Normalize
    mean = norm_params['means'][neuron_id]
    std = norm_params['stds'][neuron_id]
    normalized = (processed_voltage - mean) / std
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Raw voltage
    start_idx, end_idx = time_window
    ax1.plot(times[start_idx:end_idx], raw_voltages[neuron_id, start_idx:end_idx], 
             'b-', linewidth=1.5)
    ax1.axhline(-50, color='red', linestyle='--', alpha=0.5, label='Spike threshold')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_title(f'Raw Voltage - Neuron {neuron_id}, Run {run_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Normalized voltage
    adj_start = start_idx - discard_ms
    adj_end = end_idx - discard_ms
    if adj_start >= 0 and adj_end < len(normalized):
        norm_times = times[start_idx:end_idx]
        ax2.plot(norm_times, normalized[adj_start:adj_end], 'g-', linewidth=1.5)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Normalized voltage')
        ax2.set_title(f'Processed & Normalized Voltage')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(preprocessed_dir, 'raw_vs_processed.png'), dpi=150)
    plt.show()


if __name__ == "__main__":
    # Run all validation functions
    print("="*60)
    print("DATA VALIDATION AND ANALYSIS")
    print("="*60)
    
    # First check if preprocessed data exists
    if os.path.exists('preprocessed/datasets.pt'):
        print("\n1. Validating preprocessed data...")
        validate_preprocessed_data()
        
        print("\n2. Visualizing raw AND processed sequences...")
        visualize_raw_and_processed_sequences()
        
        print("\n3. Comparing voltage distributions...")
        visualize_voltage_distributions()
        
        print("\n4. Comparing single neuron raw vs processed...")
        compare_raw_and_processed()
    else:
        print("No preprocessed data found. Run preprocess_data.py first!")
    
    # Analyze spike statistics (works with raw data)
    if os.path.exists('output_raw'):
        print("\n6. Analyzing spike statistics...")
        analyze_spike_statistics()