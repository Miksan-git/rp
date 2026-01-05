# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation Steps

1. **Navigate to project directory**
```bash
cd /Users/miksan/Desktop/rp
```

2. **Create virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Running the Project

### Step 1: Analyze the Dataset

First, understand your data:
```bash
python notebooks/data_analysis.py
```

This will generate:
- Dataset statistics
- Feature distributions
- Visualization plots in `results/analysis/`

### Step 2: Train the Model

Train the model with default settings:
```bash
python src/train.py
```

**What happens:**
- Data is loaded and preprocessed
- Train/Val/Test split (70/15/15)
- Model architecture is created
- Training starts with progress bars
- Best model is saved automatically
- Evaluation metrics are printed

**Output:**
- `models/saved/best_model.pth` - Trained model
- `models/saved/preprocessor.pkl` - Preprocessing pipeline
- `results/confusion_matrix.png` - Confusion matrix
- `logs/` - TensorBoard logs

**Monitor training:**
```bash
tensorboard --logdir logs
```
Open http://localhost:6006 in your browser

### Step 3: Make Predictions

Use the trained model for inference:

```python
from src.inference import TreatmentPredictor

# Initialize predictor
predictor = TreatmentPredictor(
    model_path="models/saved/best_model.pth",
    preprocessor_path="models/saved/preprocessor.pkl",
    config_path="configs/config.yaml"
)

# Predict for a single case
result = predictor.predict_single(
    breed="Chihuahua",
    age=6,
    weight=2.8,
    medical_history="None",
    genetic_predispositions="None",
    current_medications="None",
    diet="Balanced",
    lifestyle="Indoor",
    environment="Temperate",
    vaccination_status="Not Up-to-date",
    neutering_status="Neutered",
    living_conditions="Multi-pet",
    disease="Hypersensitivity Allergic Dermatosis",
    stage="Severe"
)

print(f"Conventional Treatment: {result['conventional_treatment']}")
print(f"Natural Remedies: {result['natural_remedies']}")
```

## Configuration

Edit `configs/config.yaml` to customize:

### Model Architecture
```yaml
model:
  embedding_dim: 32        # Embedding dimension
  hidden_layers: [256, 128, 64]  # Shared layers
  dropout_rate: 0.3         # Dropout probability
```

### Training Parameters
```yaml
training:
  batch_size: 64           # Batch size
  num_epochs: 100          # Maximum epochs
  learning_rate: 0.001     # Initial learning rate
  early_stopping_patience: 10  # Early stopping patience
```

### Data Splits
```yaml
data:
  train_split: 0.7         # Training set proportion
  val_split: 0.15         # Validation set proportion
  test_split: 0.15        # Test set proportion
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - Solution: Make sure virtual environment is activated and dependencies are installed

2. **CUDA out of memory**
   - Solution: Reduce `batch_size` in `configs/config.yaml`

3. **FileNotFoundError for dataset**
   - Solution: Check that `Refined_Book_Aligned_Dog_Treatment_Dataset.csv` is in the project root

4. **Import errors**
   - Solution: Run from project root directory, or add project to PYTHONPATH

## Expected Training Time

- **Dataset size**: ~50,000 samples
- **Training time**: ~30-60 minutes on CPU, ~5-10 minutes on GPU
- **Epochs**: Usually converges in 20-40 epochs with early stopping

## Next Steps

1. **Experiment with architecture**: Modify `configs/config.yaml`
2. **Hyperparameter tuning**: Try different learning rates, batch sizes
3. **Feature engineering**: Add new features or modify existing ones
4. **Model interpretation**: Add explainability features
5. **Mobile integration**: Export model for Flutter app

## Project Structure Reference

```
rp/
├── configs/config.yaml          # Configuration
├── src/
│   ├── train.py                 # Training script
│   ├── inference.py             # Inference script
│   ├── data/                    # Data processing
│   ├── models/                  # Model architecture
│   └── evaluation/              # Metrics
├── notebooks/
│   └── data_analysis.py         # Data analysis
├── models/saved/                # Saved models
├── logs/                        # TensorBoard logs
└── results/                     # Results and plots
```

## Getting Help

- Check `README.md` for detailed documentation
- Check `ARCHITECTURE.md` for technical details
- Review code comments in source files
- Check TensorBoard logs for training curves

