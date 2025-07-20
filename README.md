# organoid-activity


## Data
- Sample data for one run is included in `data/output` (`run_0_spikes.dat`, `run_0_voltages.dat`).
- Full dataset is (50 runs, ~400 MB)

## Important Notes on Simulation Parameters

### Neural Activity Characteristics

The simulated neural network in this project exhibits the following characteristics:

- **Firing Rate**: ~30 Hz (slightly elevated compared to typical cortical neurons at 5-20 Hz)
- **Regularity**: CV_ISI ~0.41 (more regular than biological neurons which typically show CV_ISI ~0.8-1.2)
- **Network Size**: 100 neurons (80 excitatory, 20 inhibitory) - smaller than typical cortical microcircuits

These parameters were chosen to balance computational efficiency with network dynamics suitable for demonstrating the deep learning pipeline. While the activity patterns are more regular than typically observed in biological neural networks, they provide stable, reproducible data for proof-of-concept development.

### Limitations and Future Directions

1. **Spiking Regularity**: The low coefficient of variation (CV_ISI = 0.41) indicates quasi-regular spiking rather than the irregular patterns characteristic of cortical neurons. This simplified dynamics may not fully capture the complexity of biological neural activity.

2. **Network Scale**: The 100-neuron network is significantly smaller than biological neural circuits. Larger networks (1000+ neurons) would better approximate the collective dynamics of neural populations.

3. **Parameter Regime**: The network operates in a regime between asynchronous irregular (AI) and synchronous regular (SR) states, rather than the pure AI state typical of cortex.

### Implications for Model Development

Despite these limitations, this controlled environment serves its intended purpose:
- Validates the complete ML pipeline from data generation through evaluation
- Establishes baseline performance metrics for temporal prediction
- Provides a foundation for transitioning to real biological data

When applying this pipeline to real neural recordings (e.g., from MEAs or calcium imaging), expect:
- Lower firing rates with higher variability
- More complex temporal dependencies
- Need for additional preprocessing and noise handling
- Potential adjustments to model architecture and hyperparameters

### Voltage Dynamics

The membrane voltage traces show expected dynamics:
- Resting potential: -70 mV (biologically accurate)
- Spike threshold: -50 mV (physiologically plausible)
- Subthreshold fluctuations: 5-15 mV range (appropriate scale)
- Reset dynamics: Proper return to resting potential after spiking

The voltage normalization (z-score per neuron) preserves relative dynamics while standardizing inputs for stable neural network training.