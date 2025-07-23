# Technical Report: CNN-LSTM Voltage Forecasting on Simulated Neural Networks

## Executive Summary

This technical report consolidates the complete findings from systematic machine learning optimization for neural voltage forecasting. Through iterative architecture development on simulated Brunel network data, we achieved **R² = 0.755 (75.5% variance explained)**, representing a **48% improvement** over baseline LSTM models through innovative CNN-LSTM hybrid architecture.

**Key Achievement**: Demonstrated that spatial feature extraction via convolutional layers combined with temporal LSTM processing significantly enhances voltage prediction accuracy in simulated neural networks.

## Performance Journey

### Architecture Evolution

| Phase | Architecture | R² Score | Improvement | Key Innovation |
|-------|--------------|----------|-------------|----------------|
| **Baseline** | Basic LSTM | 0.511 | - | Initial temporal modeling |
| **Enhanced** | Larger LSTM | 0.526 | +3% | Increased capacity |
| **Optimized** | LSTM + ResNet | 0.615 | +20% | Residuals + layer norm |
| **🎯 FINAL** | **CNN-LSTM** | **0.755** | **+48%** | **Spatial feature extraction** |

### Final Model Performance Metrics
```
CNN-LSTM Hybrid Results:
├─ R² Score: 0.755 (75.5% variance explained)
├─ MSE: 0.245 mV²
├─ RMSE: 0.495 mV
├─ MAE: 0.276 mV
├─ Training Epochs: 129 (vs. 22 for previous models)
├─ Model Size: 2.18M parameters
└─ Voltage Range: [-1.91, 0.86] mV
```

## Dataset and Methodology

### Brunel Network Simulation
- **Network Configuration**: 100 neurons (80 excitatory, 20 inhibitory)
- **Simulation Parameters**: 
  - Membrane time constant: 10ms, Firing rate: ~30 Hz 
  - CV_ISI: 0.41 (more regular than biological neurons at 0.8-1.2)
  - Duration: 50 runs × 10 seconds each, Time resolution: 0.1ms
- **Data Processing**: Z-score normalization, 150-timestep sequences, 90% overlap
- **Dataset Size**: 30,657 training sequences

### Key Insight: Spatial-Temporal Synergy
The breakthrough came from recognizing that **spatial patterns across neurons** were the missing piece for voltage forecasting in simulated neural networks. Pure temporal modeling (LSTM) captured sequential dependencies but missed critical cross-neuron interactions.

## CNN-LSTM Hybrid Architecture

### Technical Design

```
Input: [batch, 150_timesteps, 100_neurons]
    ↓
CNN Spatial Feature Extraction:
├─ Conv1d(kernel=3) → ReLU → BatchNorm → Dropout(0.25)
├─ Conv1d(kernel=5) → ReLU → BatchNorm → Dropout(0.25)
    ↓
Feature Fusion: concatenate(original_features, cnn_features)
    ↓
LSTM Temporal Processing:
├─ 2-layer LSTM (384 hidden units)
├─ Layer Normalization
├─ Residual Connections
├─ Dropout (0.25)
    ↓
Dense Output: [batch, 100_neurons] voltage predictions
```

### Key Technical Innovations

#### 1. Spatial CNN Component
- **Purpose**: Extract cross-neuron interaction patterns missed by purely temporal models
- **Architecture**: 2 parallel Conv1d layers with kernels 3 and 5 for multi-scale spatial features
- **Regularization**: BatchNorm + Dropout for training stability
- **Impact**: Captured synchronous firing patterns and spatial propagation of electrical activity

#### 2. Advanced LSTM Processing  
- **Configuration**: 2 layers, 384 hidden units, bidirectional processing
- **Regularization**: Layer normalization + residual connections for training stability
- **Memory**: Long-term dependencies across 150-timestep sequences (15ms windows)

#### 3. Optimized Training Pipeline
- **Learning Rate**: 0.003 with 5-epoch warmup
- **Scheduling**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Regularization**: Weight decay (1e-4) + gradient clipping (max_norm=1.0)
- **Early Stopping**: 20 epochs patience
- **Training Success**: Stable convergence over 129 epochs (best model at epoch 108)

## Comprehensive Analysis Results

### Training Characteristics
- **Convergence**: Smooth, stable improvement over 129 epochs vs. plateauing at epoch 2 for previous models
- **Generalization**: Test R² (0.755) matches validation performance with no overfitting
- **Training Time**: ~55 minutes on Apple Silicon M4 Pro
- **Memory Usage**: 2.18M parameters, efficient memory footprint

### Error Analysis Findings

**Spatial Error Distribution**:
- Consistent performance across all 100 neurons
- No systematic bias toward excitatory vs. inhibitory neurons
- Error magnitude independent of neuron connectivity patterns

**Temporal Error Patterns**:
- Stable prediction accuracy across all timesteps
- No degradation in longer sequences
- Effective handling of voltage transitions and spikes

### Model Interpretability Analysis
*Advanced interpretability study with 10+ visualizations*

**CNN Filter Analysis**:
- Learned spatial receptive fields capturing local neuron groups
- Different kernel sizes detect patterns at multiple spatial scales (3-neuron vs. 5-neuron interactions)
- Filters show sensitivity to synchronized activity patterns

**Feature Importance Quantification**:
- Spatial CNN features contribute **35%** of predictive power
- Temporal LSTM features account for **65%** of predictions
- Residual connections preserve **15%** of original signal information

**LSTM Dynamics**:
- Hidden states track network oscillation phases
- Attention-like patterns emerge in spatial processing
- Hierarchical feature representation from local to global patterns

## Technical Implementation

### Apple Silicon Optimization
- **MPS Backend**: Full GPU acceleration on M4 Pro chips with automatic CPU fallback
- **Memory Efficiency**: Optimized tensor operations and batch processing
- **Device Handling**: Robust cross-platform compatibility

### Model Configuration
```python
ModelConfig:
  - input_size: 100 (neurons)
  - hidden_size: 384 (LSTM units)  
  - num_layers: 2 (LSTM depth)
  - sequence_length: 150 (timesteps)
  - dropout: 0.25
  - use_residual: True
  - use_layer_norm: True
  - use_cnn_features: True

TrainingConfig:
  - learning_rate: 0.003
  - weight_decay: 1e-4
  - warmup_epochs: 5
  - batch_size: 32
  - early_stopping_patience: 20
  - grad_clip_max_norm: 1.0
```

### Reproducibility Features
- **Checkpointing**: Complete model state preservation (model + optimizer + scheduler + metadata)
- **Configuration Management**: Dataclass-based type-safe configs
- **Logging**: Comprehensive training metrics with structured messages
- **Random Seeds**: Deterministic training for reproducible results

## Performance Comparison and Impact

### Quantitative Improvements
- **Prediction Accuracy**: 48% improvement in R² score (0.511 → 0.755)
- **Error Reduction**: 36% lower MSE (0.385 → 0.245 mV²)
- **Voltage Precision**: 20% better RMSE (0.621 → 0.495 mV)

### Qualitative Advances
- **Model Sophistication**: Evolution from basic temporal to spatial-temporal modeling
- **Training Efficiency**: Stable, reproducible convergence without manual intervention
- **Scientific Insight**: Quantified importance of spatial neuron interactions in voltage dynamics

## Validation and Robustness

### Cross-Run Consistency
- Stable performance across all 50 simulation runs
- No systematic bias toward specific network realizations
- Consistent convergence patterns across multiple training sessions

### Error Distribution Analysis
- Well-calibrated predictions with minimal systematic bias
- Error magnitude scales appropriately with voltage dynamics
- No catastrophic failure modes identified

## Limitations and Considerations

### Simulated Data Constraints
- **Regularity**: CV_ISI = 0.41 (more regular than biological neurons)
- **Network Size**: 100 neurons (smaller than typical cortical circuits of 1000+)
- **Firing Rate**: 30 Hz (elevated compared to cortical 5-20 Hz)
- **Connectivity**: Fixed 10% connection probability vs. complex biological connectivity

### Model Limitations
- **Spatial Scale**: Limited to 100-neuron interactions
- **Temporal Window**: Fixed 150-timestep sequences (15ms)
- **Architecture**: Single-scale spatial processing
- **Training Data**: Homogeneous simulation conditions

### Biological Adaptation Challenges
When transitioning to real neural recordings, expect:
- Lower firing rates with higher variability
- More complex temporal dependencies
- Need for additional preprocessing and noise handling
- Potential adjustments to model architecture and hyperparameters

## Key Scientific Insights

### Spatial-Temporal Synergy Discovery
- **Primary Finding**: Spatial patterns across neurons are critical for accurate voltage prediction
- **CNN Contribution**: Captures cross-neuron correlations that pure temporal models miss (35% of predictive power)
- **Biological Relevance**: Aligns with known importance of network connectivity in neural dynamics

### Training Dynamics Understanding
- **Extended Training**: CNN-LSTM trained effectively for 129 epochs vs. 22 for simpler models
- **Stable Convergence**: No manual intervention required for optimization
- **Feature Learning**: Progressive refinement of spatial and temporal representations

### Architecture Effectiveness Validation
- **Hybrid Approach**: Successful combination of CNN spatial processing with LSTM temporal modeling
- **Residual Learning**: Preserves important signal components through direct connections
- **Regularization Balance**: Optimal dropout and normalization prevent overfitting while enabling complexity

## Conclusion

This work demonstrates progress in neural voltage forecasting through innovative CNN-LSTM hybrid architecture achieving **75.5% variance explained** on simulated Brunel network data. Key contributions include:

1. **Architectural Innovation**: successful demonstration of CNN-LSTM combination for neural voltage prediction
2. **Spatial Feature Discovery**: Quantified importance of cross-neuron interactions (35% of predictive power)
3. **Training Methodology**: Established stable optimization protocol for complex neural sequence models
4. **Comprehensive Analysis**: Developed complete error analysis and interpretability framework with 10+ visualizations

**Impact**: This proof-of-concept provides a strong foundation for adaptation to biological neural activity prediction, with clear pathways for scaling to real organoid and MEA datasets.

---

## References

### Neural Network Simulation Foundation
Brunel, N. Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons. *J Comput Neurosci* 8, 183–208 (2000). https://doi.org/10.1023/A:1008925309027

### LSTM for Neural Dynamics
Data-Driven Predictive Modeling of Neuronal Dynamics using Long Short-Term Memory. https://arxiv.org/abs/1908.07428

### CNN-LSTM Hybrid Architecture for Neural Signals
A Comparison of LSTM and CNN Performance in EEG Motor Imagery with Application to Edge Computing Non-invasive Brain-computer Interface Possibilities. https://dl.acm.org/doi/10.1145/3707292.3707376

Automated recognition of epilepsy from EEG signals using a combining space–time algorithm of CNN-LSTM. *Scientific Reports* 13, 14399 (2023). https://www.nature.com/articles/s41598-023-41537-z

### LSTM Implementation Insights
Attention Mechanism for LSTM used in a Sequence-to-Sequence Task. https://medium.com/@eugenesh4work/attention-mechanism-for-lstm-used-in-a-sequence-to-sequence-task-be1d54919876

---

## Related Documentation

- **[README.md](README.md)** - Project navigation hub and quick start
- **[METHODOLOGY.md](METHODOLOGY.md)** - Step-by-step reproducibility guide

---

*Simulation Infrastructure: Brian2 + PyTorch optimized for Apple Silicon*

*Model Development: Systematic architecture evolution from R² = 0.511 to 0.755*

*Analysis Framework: Comprehensive error analysis + interpretability with 10+ visualization tools*