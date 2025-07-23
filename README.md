# Organoid Voltage Forecasting

**CNN-LSTM hybrid model achieving R² = 0.755 (48% improvement) for voltage prediction on simulated Brunel network data.** This project demonstrates effective spatial-temporal modeling for neural voltage forecasting using deep learning on 100-neuron simulated datasets.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run model inference demo
python -m src.predict_example

# Generate analysis visualizations
python -m src.error_analysis
python -m src.interpretability
```

## Navigation

### 📊 Core Documentation
- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** - Complete findings consolidation
- **[METHODOLOGY.md](METHODOLOGY.md)** - Step-by-step reproducibility guide

### 📈 Analysis & Results
- **src/analysis_results/** - Error analysis visualizations and metrics
- **src/interpretability_results/** - Model interpretability analysis

### 🔧 Implementation
- **src/** - Core source code (models, training, analysis)
- **models/** - Trained model checkpoints and results
- **data/** - Simulated Brunel network datasets

## Project Structure

```
organoid_activity/
├── README.md                    # This navigation hub
├── TECHNICAL_REPORT.md          # Consolidated findings
├── METHODOLOGY.md               # Reproducibility guide
├── src/                        # Source code
│   ├── lstm_forecaster.py      # CNN-LSTM model architecture
│   ├── trainer.py              # Training pipeline
│   ├── error_analysis.py       # Error analysis tools
│   ├── interpretability.py     # Model interpretability
│   └── analysis_results/       # Generated visualizations
├── models/                     # Model checkpoints
│   └── best_model.pth          # Final trained model (R² = 0.755)
├── data/                       # Simulated datasets
│   └── preprocessed/           # Processed training data
└── notebooks/                  # Development notebooks
```

## Key Results

- **Architecture**: CNN-LSTM hybrid with spatial feature extraction
- **Performance**: R² = 0.755 (75.5% variance explained)
- **Dataset**: 100-neuron Brunel network simulation (80 excitatory, 20 inhibitory)
- **Training**: 129 epochs, stable convergence, Apple Silicon optimized
- **Analysis**: Comprehensive error analysis + interpretability study with 10+ visualizations

## Available Commands

```bash
# Core functionality  
python -m src.predict_example       # Model usage demonstration
python -m src.train_model          # Retrain model from scratch

# Analysis tools
python -m src.error_analysis        # Generate error analysis
python -m src.interpretability      # Model interpretability analysis

# Visualization generators
python -m src.error_visualizer      # Error visualization dashboard
python -m src.interpretability_viz  # Interpretability visualizations
```

## Data Notes

**Simulated Data**: This project uses controlled Brunel network simulations with 30 Hz firing rates and CV_ISI ~0.41 (more regular than biological neurons). The simplified dynamics enable stable proof-of-concept development while maintaining essential neural network characteristics.

**Limitations**: 100-neuron network scale, regular spiking patterns, and controlled environment. See "Next Phase" section below for biological data adaptation strategies.

---

## Next Phase: Biological Data Integration

### Multi-Electrode Array (MEA) Adaptation
**Objective**: Adapt CNN-LSTM for high-density MEA recordings from neural cultures and organoids.

**Key Challenges**:
- **Noise Handling**: Real MEA data has 10-100x more noise than simulations
- **Spatial Geometry**: Incorporate actual electrode positions for spatial convolutions
- **Variable Sampling**: Different MEA systems use 10-40 kHz sampling rates

**Technical Approach**:
- Adaptive preprocessing with robust artifact detection
- Spatial-aware CNN respecting electrode geometry
- Transfer learning from Brunel-trained weights to MEA data

### Calcium Imaging Integration
**Objective**: Extend to optical neural activity measurements with different temporal dynamics.

**Adaptations**:
- Handle 30-100 Hz temporal resolution (vs. 10,000 Hz for voltage)
- Process calcium transients instead of membrane voltage
- Manage 1000+ neurons vs. 64-256 electrodes
- Address photon noise, motion artifacts, and bleaching

### Multi-Modal Data Fusion
**Vision**: Combine MEA electrical recordings with calcium imaging for comprehensive neural activity modeling.

---

*This is a proof-of-concept implementation demonstrating CNN-LSTM voltage forecasting on simulated neural data. For complete technical details, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).*