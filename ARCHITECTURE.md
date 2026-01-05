# Architecture Documentation

## System Architecture Overview

This document provides detailed technical documentation of the machine learning architecture for the Dog Treatment Recommendation System.

## 1. Problem Formulation

### Task Definition
- **Input**: Mixed categorical and numerical features describing a dog's profile and disease condition
- **Output 1**: Conventional veterinary treatment (single-label classification)
- **Output 2**: Natural/home-remedy treatments (multi-label classification)

### Why This Architecture?

1. **Multi-Task Learning Benefits**:
   - Shared representation learning improves generalization
   - Reduces overfitting by learning common patterns
   - More efficient than training separate models

2. **Embedding-Based Categorical Encoding**:
   - Learns meaningful representations for categorical features
   - Better than one-hot encoding (reduces dimensionality)
   - Captures relationships between categories

3. **Separate Output Heads**:
   - Different tasks require different output activations
   - Single-label: Softmax (mutually exclusive)
   - Multi-label: Sigmoid (independent binary predictions)

## 2. Architecture Components

### 2.1 Input Processing

#### Categorical Features
```
Feature: Breed (e.g., "Chihuahua")
  ↓
Label Encoding: "Chihuahua" → 5
  ↓
Embedding Lookup: Embedding(5) → [32-dim vector]
```

Each categorical feature has:
- Unique vocabulary size (num_classes)
- Dedicated embedding layer (embedding_dim = 32)
- Learned representation during training

**Total Categorical Embedding Dimension**: 
`num_categorical_features × embedding_dim = 12 × 32 = 384`

#### Numerical Features
```
Feature: Age, Weight
  ↓
StandardScaler: (x - μ) / σ
  ↓
Normalized values: [Age_norm, Weight_norm]
```

**Total Numerical Dimension**: 2

#### Combined Input
```
Combined = [Categorical_Embeddings (384-dim), Numerical_Features (2-dim)]
Total Input Dimension = 386
```

### 2.2 Shared Representation

The shared layers learn common patterns useful for both tasks:

```
Input (386-dim)
  ↓
Dense(256) + BatchNorm + Dropout(0.3) + ReLU
  ↓
Dense(128) + BatchNorm + Dropout(0.3) + ReLU
  ↓
Dense(64) + BatchNorm + Dropout(0.3) + ReLU
  ↓
Shared Representation (64-dim)
```

**Why Shared Layers?**
- Learns high-level features (e.g., "severe bacterial infection in young dog")
- Reduces model complexity
- Improves generalization

### 2.3 Output Heads

#### Conventional Treatment Head
```
Shared Representation (64-dim)
  ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Dense(num_conventional_classes)
  ↓
Softmax → Probability Distribution
```

**Output**: Probability distribution over treatment classes
**Prediction**: Argmax (highest probability)

#### Natural Remedies Head
```
Shared Representation (64-dim)
  ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.2)
  ↓
Dense(num_natural_classes)
  ↓
Sigmoid → Independent Probabilities
```

**Output**: Independent probabilities for each remedy
**Prediction**: Threshold at 0.5 (binary classification per remedy)

## 3. Loss Function

### Multi-Task Loss

```
L_total = w_conv × L_conv + w_nat × L_nat
```

Where:
- `L_conv = CrossEntropyLoss(conventional_logits, conventional_target)`
- `L_nat = BCEWithLogitsLoss(natural_logits, natural_target)`
- `w_conv = 1.0` (configurable)
- `w_nat = 1.0` (configurable)

### Why This Loss?

1. **CrossEntropyLoss for Conventional**:
   - Standard for single-label classification
   - Handles class imbalance (with optional class weights)

2. **BCEWithLogitsLoss for Natural**:
   - Standard for multi-label classification
   - Treats each remedy as independent binary classification
   - Numerically stable (logits before sigmoid)

## 4. Training Strategy

### Optimization
- **Optimizer**: Adam (adaptive learning rate)
- **Learning Rate**: 0.001 (configurable)
- **Weight Decay**: 0.0001 (L2 regularization)
- **Batch Size**: 64

### Learning Rate Scheduling
- **Scheduler**: ReduceLROnPlateau
- **Factor**: 0.5 (halve learning rate)
- **Patience**: 5 epochs
- **Mode**: Minimize validation loss

### Early Stopping
- **Patience**: 10 epochs
- **Min Delta**: 0.001
- **Monitor**: Validation loss
- **Mode**: Minimize

### Regularization
1. **Dropout**: 0.3 (shared layers), 0.2 (output heads)
2. **Batch Normalization**: After each dense layer
3. **Weight Decay**: L2 regularization in optimizer
4. **Early Stopping**: Prevents overfitting

## 5. Model Complexity

### Parameter Count

**Embedding Layers**:
- 12 categorical features × (avg_vocab_size × 32)
- Example: ~12 × (10 × 32) = ~3,840 parameters

**Shared Layers**:
- (386 × 256) + 256 = 98,816
- (256 × 128) + 128 = 32,896
- (128 × 64) + 64 = 8,256
- **Total**: ~140,000 parameters

**Conventional Head**:
- (64 × 64) + 64 = 4,160
- (64 × n_classes) + n_classes ≈ 320 (if 5 classes)
- **Total**: ~4,500 parameters

**Natural Head**:
- (64 × 64) + 64 = 4,160
- (64 × n_remedies) + n_remedies ≈ 640 (if 10 remedies)
- **Total**: ~4,800 parameters

**Total Model Size**: ~150,000 parameters (lightweight, suitable for mobile)

## 6. Evaluation Strategy

### Metrics Selection

**Conventional Treatment**:
- Accuracy: Overall correctness
- F1-Macro: Balanced performance across classes
- Precision/Recall: Per-class performance
- Confusion Matrix: Visual error analysis

**Natural Remedies**:
- Hamming Loss: Average label error rate
- F1-Macro/Micro: Label-wise/sample-wise F1
- Jaccard Score: Label set similarity
- Per-Label Metrics: Individual remedy performance

### Why These Metrics?

1. **Single-Label Metrics**: Standard classification metrics
2. **Multi-Label Metrics**: Account for partial correctness
3. **Macro vs Micro**: Macro (per-class average), Micro (overall)
4. **Hamming Loss**: Interpretable (fraction of wrong labels)

## 7. Scalability Considerations

### Mobile Deployment
- Model size: ~600 KB (150K parameters × 4 bytes)
- Inference time: <10ms on modern mobile CPU
- Memory: <5 MB during inference

### Future Enhancements
1. **Quantization**: INT8 quantization (4× smaller)
2. **Pruning**: Remove redundant connections
3. **Knowledge Distillation**: Smaller student model
4. **ONNX Export**: Cross-platform deployment

## 8. Research Novelty

### Academic Contributions

1. **Multi-Task Learning for Veterinary Applications**:
   - First application of multi-task learning to simultaneous conventional + natural treatment recommendation

2. **Embedding-Based Architecture for Mixed Data**:
   - Demonstrates effectiveness of embeddings for categorical veterinary features

3. **Clinical Decision Support**:
   - Practical application with real-world dataset
   - Addresses both conventional and alternative treatments

4. **Mobile-Ready Architecture**:
   - Lightweight design suitable for edge deployment
   - Enables real-time recommendations

## 9. Limitations and Future Work

### Current Limitations
1. **Static Model**: No online learning from new cases
2. **No Explainability**: Black-box predictions
3. **Limited to Dataset Diseases**: Cannot handle unseen diseases
4. **No Image Input**: Cannot analyze food images yet

### Future Enhancements
1. **Explainability**: SHAP, LIME, attention visualization
2. **RAG Integration**: Retrieve relevant knowledge for explanations
3. **Image Branch**: CNN for food image classification
4. **Continuous Learning**: Update model with new cases
5. **Uncertainty Quantification**: Confidence scores for predictions

## 10. Comparison with Alternatives

### vs. Separate Models
- **Advantage**: Shared representation, fewer parameters
- **Disadvantage**: Less task-specific optimization

### vs. One-Hot Encoding
- **Advantage**: Lower dimensionality, learned representations
- **Disadvantage**: Requires more training data

### vs. Traditional ML (Random Forest, XGBoost)
- **Advantage**: Better for complex non-linear patterns, end-to-end learning
- **Disadvantage**: Requires more data, less interpretable

## References

1. Multi-Task Learning: Caruana, R. (1997). "Multitask Learning"
2. Embedding Layers: Mikolov et al. (2013). "Efficient Estimation of Word Representations"
3. Multi-Label Classification: Zhang & Zhou (2014). "A Review on Multi-Label Learning Algorithms"

