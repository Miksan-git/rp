# AI-Driven Dog Care Mobile Application with Disease-Aware Treatment Recommendation System

## Project Overview

This is a 1-year undergraduate thesis research project that develops a multi-input, multi-output machine learning system for recommending both conventional veterinary treatments and natural/home-remedy treatments for dogs based on comprehensive dog profiles and disease information.

## Research Objectives

1. **Multi-Task Learning**: Predict both conventional treatments (single-label) and natural remedies (multi-label) simultaneously
2. **Mixed Data Handling**: Process both categorical and numerical features using embedding-based architecture
3. **Clinical Decision Support**: Provide evidence-based treatment recommendations for veterinary applications
4. **Scalability**: Design architecture suitable for mobile app integration (Flutter)

## Dataset

- **Size**: ~50,000 realistic dog cases
- **Format**: CSV with comprehensive feature set
- **Location**: `Refined_Book_Aligned_Dog_Treatment_Dataset.csv`

### Input Features

**Categorical Features:**
- Breed
- Medical History
- Genetic Predispositions
- Current Medications
- Diet
- Lifestyle (Indoor/Outdoor/Active/Mixed)
- Environment (Temperate/Tropical/Cold/Humid/Wet/Dry)
- Vaccination Status
- Neutering Status
- Living Conditions (Single-pet/Multi-pet)
- Disease
- Disease Stage (Initial/Mild/Moderate/Severe)

**Numerical Features:**
- Age (years)
- Weight (kilograms)

### Target Variables

- **Conventional Treatment**: Single-label classification (e.g., Amoxicillin, Cephalexin, Ketoconazole)
- **Natural Remedies**: Multi-label classification (comma-separated, e.g., "Chamomile, Aloe vera gel, Probiotics")

## Architecture

### Neural Network Design

```
Input Layer
├── Categorical Features → Embedding Layers (32-dim each)
├── Numerical Features → Normalized Input
│
Shared Representation
├── Dense Layer 1 (256 units) + BatchNorm + Dropout (0.3)
├── Dense Layer 2 (128 units) + BatchNorm + Dropout (0.3)
└── Dense Layer 3 (64 units) + BatchNorm + Dropout (0.3)
│
Output Heads
├── Conventional Treatment Head
│   ├── Dense (64 units) + BatchNorm + ReLU + Dropout (0.2)
│   └── Output Layer (n_classes) → Softmax
│
└── Natural Remedies Head
    ├── Dense (64 units) + BatchNorm + ReLU + Dropout (0.2)
    └── Output Layer (n_remedies) → Sigmoid
```

### Key Design Decisions

1. **Embedding Layers**: Each categorical feature gets its own embedding layer (32 dimensions) to learn meaningful representations
2. **Shared Representation**: Common layers learn shared patterns before task-specific heads
3. **Multi-Task Learning**: Joint training improves generalization by sharing learned features
4. **Regularization**: BatchNorm and Dropout prevent overfitting
5. **Activation Functions**: ReLU for hidden layers, Softmax for conventional (single-label), Sigmoid for natural (multi-label)

## Project Structure

```
rp/
├── configs/
│   └── config.yaml              # Model and training configuration
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # Data preprocessing pipeline
│   │   └── dataset.py           # PyTorch Dataset and DataLoader
│   ├── models/
│   │   ├── architecture.py     # Neural network architecture
│   │   └── losses.py            # Multi-task loss function
│   ├── evaluation/
│   │   └── metrics.py           # Evaluation metrics
│   ├── train.py                 # Training script
│   └── inference.py             # Inference script
├── notebooks/
│   └── data_analysis.py         # Dataset analysis
├── models/
│   └── saved/                   # Saved models and preprocessors
├── logs/                        # TensorBoard logs
├── results/                     # Evaluation results and plots
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Analysis

Analyze the dataset to understand distributions and data quality:

```bash
python notebooks/data_analysis.py
```

### 2. Training

Train the model with default configuration:

```bash
python src/train.py
```

The training script will:
- Load and preprocess the data
- Create train/validation/test splits
- Initialize the model architecture
- Train with early stopping
- Evaluate on test set
- Save the best model and preprocessor

**Configuration**: Edit `configs/config.yaml` to modify:
- Model architecture (embedding dimensions, hidden layers, dropout)
- Training hyperparameters (batch size, learning rate, epochs)
- Data splits
- Loss weights

### 3. Inference

Make predictions on new data:

```python
from src.inference import TreatmentPredictor

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

## Evaluation Metrics

### Conventional Treatment (Single-Label)
- **Accuracy**: Overall classification accuracy
- **F1-Score (Macro)**: Average F1 across all classes
- **Precision (Macro)**: Average precision across all classes
- **Recall (Macro)**: Average recall across all classes
- **Confusion Matrix**: Per-class performance visualization

### Natural Remedies (Multi-Label)
- **Hamming Loss**: Fraction of incorrectly predicted labels
- **F1-Score (Macro/Micro)**: Average F1 across labels/samples
- **Precision (Macro)**: Average precision across labels
- **Recall (Macro)**: Average recall across labels
- **Jaccard Score**: Intersection over union for label sets

## Dataset Size Recommendations

Based on the complexity of the task:

- **Minimum Dataset Size**: 5,000 samples
  - Sufficient for initial proof-of-concept
  - May require strong regularization
  
- **Ideal Dataset Size**: 20,000-50,000 samples
  - Good balance between data collection effort and model performance
  - Current dataset: ~50,000 samples ✅
  
- **Excellent/Publication-Level**: 100,000+ samples
  - Enables more complex architectures
  - Better generalization to rare cases
  - Suitable for publication-quality research

## Data Collection Strategy

### Manual Data Entry (First 3 Months)
1. **Veterinary Knowledge Sources**:
   - Veterinary textbooks and clinical guides
   - Peer-reviewed research papers
   - Veterinary databases (VetMed, PubMed)
   - Clinical case studies

2. **Validation Rules**:
   - Age: 0-20 years (realistic dog lifespan)
   - Weight: Breed-appropriate ranges
   - Disease-Stage consistency: Severe cases should have appropriate treatments
   - Medication-Disease compatibility: Check for contraindications
   - Breed-Disease associations: Validate known breed predispositions

3. **Quality Control**:
   - Cross-reference with multiple sources
   - Expert review (veterinary consultation)
   - Automated validation scripts
   - Regular data audits

## Data Preprocessing

1. **Missing Values**:
   - Categorical: Fill with "None"
   - Numerical: Fill with median value

2. **Categorical Encoding**:
   - Label encoding for embedding lookup
   - Each feature gets unique embedding layer

3. **Numerical Normalization**:
   - StandardScaler (zero mean, unit variance)
   - Applied to Age and Weight

4. **Multi-Label Encoding**:
   - MultiLabelBinarizer for natural remedies
   - Binary matrix: (n_samples, n_remedies)

5. **Train/Val/Test Split**:
   - 70% / 15% / 15% (stratified if possible)

## Future Enhancements

1. **Image-Based Food Analysis**: Add CNN branch for food image classification
2. **Explainability**: SHAP values, attention mechanisms, feature importance
3. **RAG (Retrieval-Augmented Generation)**: Integrate with veterinary knowledge base
4. **Mobile App Integration**: Flutter app with model inference
5. **Real-Time Updates**: Continuous learning from new cases

## Academic Justification

This project demonstrates research-level complexity through:

1. **Multi-Task Learning**: Simultaneous optimization of two different tasks
2. **Mixed Data Architecture**: Embedding-based handling of categorical + numerical features
3. **Clinical Application**: Real-world veterinary decision support system
4. **Scalability**: Architecture suitable for mobile deployment
5. **Comprehensive Evaluation**: Multiple metrics for both single-label and multi-label tasks

## Results and Performance

After training, check:
- `models/saved/best_model.pth`: Best model checkpoint
- `models/saved/preprocessor.pkl`: Preprocessing pipeline
- `results/confusion_matrix.png`: Confusion matrix visualization
- `logs/`: TensorBoard logs for training curves

View TensorBoard:
```bash
tensorboard --logdir logs
```

## Citation

If you use this code or dataset in your research, please cite:

```
AI-Driven Dog Care Mobile Application with Disease-Aware Treatment Recommendation System
[Your Name], [Your Institution], [Year]
```

## License

[Specify your license]

## Contact

[Your contact information]

## Acknowledgments

- Veterinary knowledge sources
- Dataset contributors
- Academic advisors

