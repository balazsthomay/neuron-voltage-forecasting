# Project Plan: A Foundational Pipeline for Predictive Modeling of Neural Activity

This project aims to build and validate a complete deep learning pipeline for predicting the electrophysiological activity of neural networks. By starting with a controlled, simulated environment, this work establishes a robust proof-of-concept and a foundational toolkit ready to be adapted for the complex, real-world biological data generated at companies like Cortical Labs.

## 1. High-Level Roadmap

**Phase 1: Controlled Data Generation** - Establish a reliable baseline by simulating neural activity data, creating a controlled environment to build and validate the ML pipeline.

**Phase 2: Baseline Model Architecture** - Design and implement a foundational LSTM model optimized for time-series prediction.

**Phase 3: Efficient Model Training** - Train the model on Apple Silicon hardware, focusing on stable convergence and best practices.

**Phase 4: Comprehensive Evaluation (Signal & Functional)** - Assess model performance using both standard signal-based metrics and more insightful, biologically-inspired functional metrics.

**Phase 5: Strategic Documentation & Future Work** - Document the project as a technical proof-of-concept and create a presentation that frames it as a direct precursor to tackling real-world challenges in Organoid Intelligence.

## 2. Detailed Step-by-Step Execution Plan

### Phase 1: Data Simulation and Preparation (Controlled Environment)

Generate data using Brian2 simulator.

**Simulation Strategy:**
- Create a network of 100 neurons using Leaky Integrate-and-Fire (LIF) models
- Apply realistic neurophysiological parameters (membrane time constant = 10 ms, firing threshold = -50 mV)
- Generate 50 experimental runs, each 10 seconds long, at 1 ms temporal resolution
- Ensure reproducible random seeds for consistent validation

**Data Processing Approach:**
- Extract continuous membrane voltage traces as primary features
- Extract ground-truth spike timestamps for functional validation in Phase 4
- Normalize voltage traces to standardized range for stable training
- Segment temporal data into fixed-length input sequences for sequence modeling
- Implement stratified data splitting (70% training, 15% validation, 15% test)

### Phase 2: Model Design and Development (Baseline Architecture)

Implement standard LSTM architecture using PyTorch.

**Architecture Design:**
- Input layer: sequences of membrane voltage time steps
- Core: single-layer LSTM with 64 hidden units
- Output: linear layer for next time step voltage prediction
- Enable GPU acceleration via torch.device("mps") for M4 Pro

**Training Configuration:**
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam with learning rate 0.001
- Batch size: 32
- Epochs: 50 with checkpoint saving

### Phase 3: Model Training and Optimization

This phase focuses on efficiently training the model and ensuring generalization to unseen data.

**Training Execution:**
- Run training for 50 epochs on M4 Pro GPU
- Monitor training and validation loss to detect overfitting
- Save model checkpoints after each epoch

**Hyperparameter Tuning:**
- Add dropout layer (e.g., Dropout(0.2)) if overfitting occurs
- Adjust learning rate or sequence length based on validation performance

### Phase 4: Comprehensive Model Evaluation

We will evaluate the model on two levels: its ability to replicate the raw signal and, more importantly, its ability to predict the functional behavior of the network.

**A. Signal-Level Performance:**
- Calculate MSE and RMSE on test set
- Use Matplotlib to plot predicted vs actual voltage traces
- Assess visual alignment and temporal correlation

**B. Biologically-Inspired Functional Performance:**
- Apply spike detection algorithm (voltage threshold crossing) to both ground-truth and predicted traces
- Compare functional metrics:
  - Mean firing rate (spikes/second)
  - Spike coincidence rate (percentage of predicted spikes within ±2ms of actual spikes)
  - Burst detection accuracy

**Results Analysis:**
- Synthesize findings from both evaluation levels
- Prioritize spike coincidence rate over MSE # functional accuracy more important than raw signal fidelity

### Phase 5: Documentation and Strategic Presentation

The final deliverable is not just a working model, but a compelling narrative for potential applications.

**Technical Documentation:**
- Create comprehensive report documenting each phase with clear rationale
- Explain simulation-first approach and dual-level evaluation methodology
- Include performance analysis and key insights
- Provide reproducible methodology for future extensions

**Strategic Presentation:**
- Develop concise presentation summarizing project achievements
- Frame work as foundational step toward biological neural network prediction
- Emphasize scalability and adaptability to real-world data
- Present clear pathway to industrial applications

**Future Work Vision:**
- **Sequence-to-Sequence Prediction**: Extend model to predict future activity windows, enabling network event forecasting
- **Advanced Architectures**: Explore more sophisticated models (GRUs, Transformers) for capturing long-range dependencies in biological systems
- **Real Data Application**: Deploy validated pipeline on multi-electrode array (MEA) data from biological systems to decode emergent computation

## 3. Technology Stack

**Language**: Python ecosystem for data science and machine learning
**Simulation**: Brian2 for realistic neural network modeling
**Machine Learning**: PyTorch with GPU optimization for Apple Silicon
**Data Processing**: Standard scientific computing libraries
**Visualization**: Matplotlib for result presentation and analysis

## 4. Success Metrics

**Technical Success**: Model achieves high spike coincidence rates (>70%) while maintaining low prediction error
**Methodological Success**: Pipeline demonstrates clear scalability path to biological data
**Strategic Success**: Work positions candidate as capable of tackling real-world neural computation challenges