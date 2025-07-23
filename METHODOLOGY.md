# Methodology: Reproducible CNN-LSTM Voltage Forecasting

This guide provides step-by-step instructions for reproducing the CNN-LSTM voltage forecasting results (R² = 0.755) on simulated Brunel network data.

## Environment Setup

### Prerequisites
```bash
# Activate miniconda environment
conda activate organoid-env

# Install required packages
pip install -r requirements.txt
```

### Required Dependencies
```
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
brian2>=2.5.0
pickle>=4.0
```

### Hardware Requirements
- **Recommended**: Apple Silicon (M1/M2) with MPS support
- **Alternative**: CUDA-compatible GPU or CPU fallback
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB for data and models

## Data Generation

### Step 1: Brunel Network Simulation

Generate simulated neural data using the included Brian2 simulation:

```bash
# Navigate to data directory
cd data/

# Run Brunel network simulation (generates 50 runs)
python brunel_2000_realistic.py

# Expected output:
# - data/output_raw/run_[0-49]_spikes.dat
# - data/output_raw/run_[0-49]_voltages.dat
# - data/simulation_summary.txt
```

**Simulation Parameters**:
- Network: 100 neurons (80 excitatory, 20 inhibitory)  
- Duration: 10 seconds per run × 50 runs
- Time resolution: 0.1ms timesteps
- Connectivity: 10% connection probability
- External input: 1000 Hz Poisson per neuron

### Step 2: Data Preprocessing

Process raw simulation data into training-ready format:

```bash
# Run preprocessing pipeline
python data_utils.py

# Generated files:
# - data/preprocessed/processed_data.pt
# - data/preprocessed/normalization_params.pkl
```

**Preprocessing Steps**:
1. Load voltage traces from all 50 runs
2. Remove initial 800ms transients
3. Apply per-neuron z-score normalization
4. Create overlapping sequences (150 timesteps, stride=15)
5. Split: 70% train, 15% validation, 15% test
6. Save as PyTorch tensors

## Model Training

### Step 3: Configuration Setup

The model uses dataclass-based configuration in `src/config.py`:

```python
@dataclass
class ModelConfig:
    input_size: int = 100          # Number of neurons
    hidden_size: int = 384         # LSTM hidden units
    num_layers: int = 2            # LSTM layers
    sequence_length: int = 150     # Input timesteps
    dropout: float = 0.25          # Dropout rate
    use_residual: bool = True      # Residual connections
    use_layer_norm: bool = True    # Layer normalization
    use_cnn_features: bool = True  # CNN spatial features

@dataclass  
class TrainingConfig:
    learning_rate: float = 0.003   # Base learning rate
    weight_decay: float = 1e-4     # L2 regularization
    warmup_epochs: int = 5         # Learning rate warmup
    batch_size: int = 32           # Training batch size
    early_stopping_patience: int = 20
    grad_clip_max_norm: float = 1.0
```

### Step 4: Model Training

Execute the complete training pipeline:

```bash
# Train CNN-LSTM model
python -m src.train_model

# Expected output:
# - models/best_model.pth          # Best validation model
# - models/latest_model.pth        # Final epoch model  
# - models/training_results.pt     # Training metrics
# - training.log                   # Comprehensive logs
```

**Training Process**:
1. **Initialization**: Model on MPS/CUDA device with proper seeds
2. **Warmup**: 5 epochs linear learning rate increase (0 → 0.003)
3. **Training**: Up to 200 epochs with early stopping (patience=20)
4. **Scheduling**: ReduceLROnPlateau (patience=3, factor=0.5)
5. **Monitoring**: Loss, R², RMSE tracked every epoch
6. **Checkpointing**: Best validation model automatically saved

**Expected Training Time**: 8-10 minutes on Apple Silicon M1/M2

### Step 5: Training Validation

Verify successful training completion:

```bash
# Check training logs
tail -20 training.log

# Expected final metrics:
# Epoch 129/200: train_loss=0.239, val_loss=0.248, val_r2=0.755
# Best model saved at epoch 108
# Training completed successfully
```

## Model Evaluation

### Step 6: Basic Performance Testing

Test the trained model:

```bash
# Run inference demonstration
python -m src.predict_example

# Expected output:
# Model loaded successfully
# Test R² Score: 0.755
# Test RMSE: 0.495 mV
# Prediction example generated
```

### Step 7: Comprehensive Error Analysis

Generate detailed error analysis:

```bash
# Run complete error analysis
python -m src.error_analysis

# Generated outputs:
# - src/analysis_results/analysis_summary.pkl
# - src/analysis_results/comprehensive_analysis.pkl
# - src/analysis_results/neuron_performance_heatmap.png
# - src/analysis_results/neuron_ranking.png
# - src/analysis_results/temporal_patterns.png
```

**Analysis Components**:
- Per-neuron performance metrics
- Temporal error patterns
- Spatial error distribution
- Statistical significance testing

### Step 8: Model Interpretability

Analyze learned representations:

```bash
# Generate interpretability analysis
python -m src.interpretability

# Generated outputs:
# - src/interpretability_results/interpretability_summary.pkl
# - src/interpretability_results/cnn_filters.png
# - src/interpretability_results/feature_importance.png
# - src/interpretability_results/lstm_dynamics.png
# - src/interpretability_results/spatial_patterns.png
# - src/interpretability_results/residual_analysis.png
```

**Interpretability Features**:
- CNN filter visualization
- Feature importance quantification
- LSTM hidden state dynamics
- Spatial pattern discovery

## Visualization Generation

### Step 9: Error Visualizations

Generate error analysis dashboards:

```bash
# Create error visualization dashboard
python -m src.error_visualizer

# Outputs comprehensive error analysis plots
```

### Step 10: Interpretability Visualizations

Create interpretability dashboards:

```bash
# Generate interpretability visualizations
python -m src.interpretability_viz

# Outputs model interpretation plots and analysis
```

## Verification and Validation

### Expected Results Checklist

**Model Performance**:
- [ ] Test R² ≥ 0.750 (target: 0.755)
- [ ] Test RMSE ≤ 0.500 mV (target: 0.495)
- [ ] Training convergence by epoch 130
- [ ] No overfitting (test ≈ validation performance)

**Generated Files**:
- [ ] `models/best_model.pth` (2.18M parameters)
- [ ] `models/training_results.pt` (complete metrics)
- [ ] Analysis results in `src/analysis_results/`
- [ ] Interpretability results in `src/interpretability_results/`

**Visualizations**:
- [ ] 5 error analysis plots
- [ ] 5 interpretability plots
- [ ] Training curves in logs

### Troubleshooting Common Issues

**MPS/CUDA Issues**:
```bash
# Check device availability
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Force CPU if needed
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**Memory Issues**:
```bash
# Reduce batch size in config.py
batch_size: int = 16  # Instead of 32

# Clear cache periodically
python -c "import torch; torch.mps.empty_cache()"
```

**Training Stalls**:
```bash
# Check data loading
python -c "from src.data_loader import load_data; load_data()"

# Verify preprocessing
ls -la data/preprocessed/
```

## Architecture Details

### CNN-LSTM Implementation

The model combines spatial and temporal processing:

```python
# Spatial feature extraction
conv1 = nn.Conv1d(100, 50, kernel_size=3, padding=1)
conv2 = nn.Conv1d(100, 50, kernel_size=5, padding=2)

# Temporal processing  
lstm = nn.LSTM(200, 384, num_layers=2, dropout=0.25)

# Output projection
output = nn.Linear(384, 100)
```

### Training Loop Structure

```python
for epoch in range(max_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        outputs = model(batch.input)
        loss = criterion(outputs, batch.target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Validation phase
    model.eval()
    val_metrics = evaluate_model(model, val_loader)
    scheduler.step(val_metrics['loss'])
    
    # Early stopping check
    if early_stopping.should_stop(val_metrics['loss']):
        break
```

## Data Format Specifications

### Input Data Format
```python
# Training sequences
input_shape: [batch_size, sequence_length, num_neurons]
# [32, 150, 100] for default configuration

# Target predictions  
target_shape: [batch_size, num_neurons]
# [32, 100] for next timestep prediction
```

### Model Output Format
```python
# Voltage predictions
output_shape: [batch_size, num_neurons] 
# Values in mV (normalized), requires denormalization for interpretation
```

## Reproducibility Checklist

**Environment**:
- [ ] Python 3.8+ with miniconda
- [ ] All dependencies installed via `pip install -r requirements.txt`
- [ ] MPS/CUDA device available (optional but recommended)

**Data**:
- [ ] Brunel simulation completed (50 runs)
- [ ] Data preprocessing successful
- [ ] Training/validation/test splits created

**Training**:
- [ ] Model configuration matches specifications
- [ ] Training completes without errors
- [ ] Best model saved with R² ≥ 0.750

**Analysis**:
- [ ] Error analysis generates expected outputs
- [ ] Interpretability analysis completes
- [ ] All visualizations created successfully

---

*For questions or issues, refer to training.log for detailed execution information and error debugging.*

---

## Related Documentation

- **[README.md](README.md)** - Project navigation hub and quick start
- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** - Complete findings consolidation
- **[FUTURE_ROADMAP.md](FUTURE_ROADMAP.md)** - Biological data adaptation pathway  
- **[MODEL_TRAINING_RESULTS.md](MODEL_TRAINING_RESULTS.md)** - Detailed training results
- **[project_plan.md](project_plan.md)** - Original project planning phases