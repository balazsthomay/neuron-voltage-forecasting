# Project Plan: A Foundational Pipeline for Predictive Modeling of Neural Activity

This project builds and validates a complete deep learning pipeline for predicting neural network electrophysiological activity. By starting with controlled simulation, this work establishes a robust **proof-of-concept** and foundational toolkit ready for adaptation to real-world biological data from companies like Cortical Labs.

## 1. High-Level Roadmap

**Phase 1: Controlled Data Generation** - Generate reliable baseline neural activity data using Brunel-like network simulation, creating a controlled environment for ML pipeline development and validation.
**Phase 2: Baseline Model Architecture** - Design and implement foundational LSTM model optimized for neural time-series prediction.
**Phase 3: Efficient Model Training** - Train model on Apple Silicon hardware with focus on stable convergence and best practices.
**Phase 4: Comprehensive Evaluation** - Assess performance using both signal-based metrics and biologically-inspired functional measures.
**Phase 5: Strategic Documentation** - Document work as technical proof-of-concept and create presentation framing it as precursor to real-world Organoid Intelligence challenges.

## 2. Detailed Step-by-Step Execution Plan

### Phase 1: Data Simulation and Preparation

Generate data using Brian2 simulator with Brunel-like network producing asynchronous irregular activity and rich temporal dynamics for training.

Brunel, N. Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons. J Comput Neurosci 8, 183–208 (2000). https://doi.org/10.1023/A:1008925309027

**Simulation Strategy:**
- Create 100-neuron network (80 excitatory, 20 inhibitory) using Leaky Integrate-and-Fire models
- Apply realistic neurophysiological parameters:
    - Membrane time constant: 10 ms
    - Resting potential: -70 mV
    - Reset potential: -70 mV
    - Threshold potential: -50 mV
    - Refractory period: 2 ms
    - Time step: 0.1 ms
    - Connection probability: 0.1
    - Excitatory synaptic weight: 0.5 mV
    - Inhibitory synaptic weight: -2.5 mV
    - External input: Poisson spikes at 1000 Hz per neuron
- Generate 50 experimental runs, each 10 seconds, recorded at 1 ms resolution
- Use reproducible random seeds for consistent validation

**Data Processing Pipeline:**
- Load voltage traces and spike times from simulation runs
- Remove initial transients (1000 ms) and add measurement noise (σ = 0.3 mV)
- Apply per-neuron z-score normalization to preserve relative dynamics
- Generate overlapping sequences: 100 ms windows with 10 ms stride (~45,000 sequences)
- Temporal data split: 70% training, 15% validation, 15% test
- Package as PyTorch datasets and preserve spike times for functional evaluation

### Phase 2: Model Design and Development

Implement LSTM architecture using PyTorch based on established neural prediction approaches.

Data-Driven Predictive Modeling of Neuronal Dynamics using Long Short-Term Memory. https://arxiv.org/abs/1908.07428
A Comparison of LSTM and CNN Performance in EEG Motor Imagery with Application to Edge Computing Non-invasive Brain-computer Interface Possibilities. https://dl.acm.org/doi/10.1145/3707292.3707376

**Architecture Design:**
- Input: 100 timesteps voltage history × 100 neurons
- Output: Next timestep voltage for all 100 neurons
- Architecture: 2 LSTM layers (256 units each), dropout 0.3, dense output layer

**Training Configuration:**
- 200 epochs, batch_size=32, MSE loss, Adam optimizer (lr=0.001)
- Early stopping (patience=10), ReduceLROnPlateau scheduler
- Gradient clipping (max_norm=1.0), num_workers=0 for MPS compatibility

### Phase 3: Model Training and Optimization

Focus on efficient training and generalization to unseen data.

**Training Execution:**
- Run 200-epoch training on M4 Pro GPU
- Monitor training and validation loss for overfitting detection
- Save model checkpoints after each epoch

**Hyperparameter Tuning:**
- Adjust learning rate or sequence length based on validation performance

### Phase 4: Comprehensive Model Evaluation

Evaluate LSTM performance across multiple dimensions, prioritizing biologically-relevant metrics over raw signal fidelity.

**Evaluation Framework:**
- **Signal-Level Performance**: RMSE, correlation analysis, spectral preservation
- **Functional Performance** *(Primary Focus)*: Spike detection accuracy, firing rate preservation, burst analysis, spike coincidence rates
- **Network Dynamics**: Population synchrony, cross-neuron correlations, emergent behaviors
- **Robustness Testing**: Noise tolerance, temporal generalization, error pattern analysis
- **Baseline Comparisons**: Performance vs. persistence model and linear autoregression with statistical significance testing

**Key Principle**: Prioritize spike coincidence rates and functional accuracy over perfect voltage reproduction—biological plausibility matters more than raw signal metrics.

**Deliverables**: Comprehensive evaluation dashboard, statistical report with confidence intervals, and comparative analysis highlighting model's ability to capture neural network dynamics.

### Phase 5: Documentation and Strategic Presentation

Deliver not just a working model, but a compelling narrative for potential applications.

**Technical Documentation:**
- Comprehensive report documenting each phase with clear rationale
- Explain simulation-first approach and dual-level evaluation methodology
- Include performance analysis and key insights
- Provide reproducible methodology for future extensions
- Methodological comparison: Why LSTM over other approaches
- Failure analysis: What didn't work and lessons learned
- Computational benchmarks for reproducibility

**Strategic Presentation:**
- Concise presentation summarizing project achievements
- Frame work as foundational step toward biological neural network prediction
- Emphasize scalability and adaptability to real-world data
- Present clear pathway to industrial applications

**Future Work Vision:**
- **Sequence-to-Sequence Prediction**: Extend model to predict future activity windows, enabling network event forecasting
- **Advanced Architectures**: Explore sophisticated models (Transformers) for capturing long-range dependencies in biological systems
- **Real Data Application**: Deploy validated pipeline on MEA data from specific systems (hippocampal slices, cortical organoids) to decode computation

## 3. Technology Stack

**Language**: Python ecosystem for data science and machine learning
**Simulation**: Brian2 for realistic neural network modeling
**Machine Learning**: PyTorch with GPU optimization for Apple Silicon
**Data Processing**: Standard scientific computing libraries
**Visualization**: Matplotlib for result presentation and analysis

## 4. Success Metrics

**Technical Success**: Model achieves >70% spike coincidence rates with controlled simulation data, demonstrating viability of LSTM approaches for neural activity prediction
**Methodological Success**: Complete pipeline demonstrates robust preprocessing, training, and evaluation workflows directly adaptable to biological MEA data
**Strategic Success**: Work establishes proof-of-concept foundation and demonstrates capability to tackle real-world neural computation challenges in commercial biotech environments