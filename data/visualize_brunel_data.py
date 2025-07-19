import numpy as np
import matplotlib.pyplot as plt
import os

# Set paths
output_dir = 'output'  # Input data subfolder
vis_dir = 'visualizations'  # Output visualization subfolder
os.makedirs(vis_dir, exist_ok=True)  # Create visualizations subfolder if it doesn't exist

# Load data for a specific run (e.g., run_0)
run_id = 0
spike_data = np.loadtxt(f'{output_dir}/run_{run_id}_spikes.dat')
voltage_data = np.loadtxt(f'{output_dir}/run_{run_id}_voltages.dat')

# Extract spike times and neuron indices
spike_times = spike_data[:, 0]  # Time in ms
spike_neurons = spike_data[:, 1]  # Neuron indices

# Extract voltage data
times = voltage_data[:, 0]  # Time in ms
voltages = voltage_data[:, 1:]  # Voltages for 100 neurons in mV

# Select subset of neurons for voltage plot (5 excitatory, 5 inhibitory)
neuron_indices = [0, 20, 40, 60, 79, 80, 85, 90, 95, 99]  # 0-79 excitatory, 80-99 inhibitory

# Compute population firing rate (spikes/second in 10 ms bins)
bin_size = 10  # ms
bins = np.arange(0, max(spike_times) + bin_size, bin_size)
spike_counts, _ = np.histogram(spike_times, bins=bins)
firing_rate = spike_counts / (bin_size / 1000) / 100  # Spikes/s/neuron
bin_centers = (bins[:-1] + bins[1:]) / 2

# Create figure with three subplots
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(nrows=3, hspace=0.4, height_ratios=[2, 2, 1])

# Raster plot
ax_raster = fig.add_subplot(gs[0])
ax_raster.scatter(spike_times, spike_neurons, s=2, color='black')
ax_raster.set_ylabel('Neuron Index')
ax_raster.set_title(f'Run {run_id} - Raster Plot')
ax_raster.set_xlim(0, 10000)  # 10 seconds in ms

# Voltage traces
ax_voltages = fig.add_subplot(gs[1], sharex=ax_raster)
for idx in neuron_indices:
    label = 'Excitatory' if idx < 80 else 'Inhibitory'
    ax_voltages.plot(times, voltages[:, idx], label=label if idx in [0, 80] else None)
ax_voltages.set_ylabel('Membrane Potential (mV)')
ax_voltages.set_title('Voltage Traces')
ax_voltages.legend()

# Population firing rate
ax_rate = fig.add_subplot(gs[2], sharex=ax_raster)
ax_rate.plot(bin_centers, firing_rate, color='blue')
ax_rate.set_xlabel('Time (ms)')
ax_rate.set_ylabel('Firing Rate (Hz)')
ax_rate.set_title('Population Firing Rate')

# Save visualization
plt.savefig(f'{vis_dir}/run_{run_id}_visualization.png', dpi=300, bbox_inches='tight')
plt.close()