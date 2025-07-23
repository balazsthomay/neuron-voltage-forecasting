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
- **[MODEL_TRAINING_RESULTS.md](MODEL_TRAINING_RESULTS.md)** - Training results and architecture details
- **[METHODOLOGY.md](METHODOLOGY.md)** - Step-by-step reproducibility guide
- **[FUTURE_ROADMAP.md](FUTURE_ROADMAP.md)** - Biological data adaptation pathway

### 📈 Analysis & Results
- **[project_plan.md](project_plan.md)** - Original project planning and phases
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
├── FUTURE_ROADMAP.md           # Next steps
├── MODEL_TRAINING_RESULTS.md   # Training details
├── project_plan.md             # Original planning
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

**Limitations**: 100-neuron network scale, regular spiking patterns, and controlled environment. See [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) for biological data adaptation strategies.

---

*This is a proof-of-concept implementation demonstrating CNN-LSTM voltage forecasting on simulated neural data. For complete technical details, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).*