# Algorithms and Techniques Used in Model Training

## 🧠 Core Machine Learning Approach

### **1. Deep Neural Network (Multi-Input, Multi-Output Architecture)**

**Type**: Feedforward Neural Network with Embeddings

**Architecture**:
```
Input Layer
├── Categorical Features → Embedding Layers (64-dim each)
├── Numerical Features → Direct Input
│
Shared Representation Layers
├── Dense Layer 1: 512 neurons + BatchNorm + Dropout(0.4) + ReLU
├── Dense Layer 2: 256 neurons + BatchNorm + Dropout(0.4) + ReLU
├── Dense Layer 3: 128 neurons + BatchNorm + Dropout(0.4) + ReLU
├── Dense Layer 4: 64 neurons + BatchNorm + Dropout(0.4) + ReLU
│
Output Heads (Multi-Task Learning)
├── Conventional Treatment Head (Single-label Classification)
│   ├── Dense: 128 neurons + BatchNorm + ReLU + Dropout(0.3)
│   └── Output: 5 classes (Softmax via CrossEntropy)
│
└── Natural Remedies Head (Multi-label Classification)
    ├── Dense: 64 neurons + BatchNorm + ReLU + Dropout(0.2)
    └── Output: 10 classes (Sigmoid for binary classification)
```

---

## 📊 Data Preprocessing Algorithms

### **1. Label Encoding**
- **Algorithm**: Scikit-learn `LabelEncoder`
- **Purpose**: Convert categorical strings to numerical indices
- **Applied to**: All categorical features (Breed, Disease, Stage, etc.)

### **2. Standard Scaling (Z-score Normalization)**
- **Algorithm**: `StandardScaler` from Scikit-learn
- **Formula**: `z = (x - μ) / σ`
- **Purpose**: Normalize numerical features (Age, Weight, Severity Score, etc.)
- **Why**: Ensures all features are on the same scale for neural network training

### **3. Multi-Label Binarization**
- **Algorithm**: `MultiLabelBinarizer` from Scikit-learn
- **Purpose**: Convert comma-separated natural remedies into binary vectors
- **Example**: `"Chamomile, Aloe vera gel"` → `[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]`

---

## 🎯 Loss Functions

### **1. Focal Loss (for Conventional Treatment)**
- **Paper**: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
- **Formula**: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`
- **Parameters**:
  - `α_t`: Class weights (handles class imbalance)
  - `γ = 3.0`: Focusing parameter (down-weights easy examples)
  - `label_smoothing = 0.1`: Prevents overconfidence
- **Why Used**: Handles severe class imbalance (Cephalexin 49% vs Ketoconazole 5%)

### **2. Binary Cross-Entropy Loss (for Natural Remedies)**
- **Algorithm**: `BCEWithLogitsLoss` with positive class weights
- **Formula**: `BCE = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]`
- **Why Used**: Multi-label classification (each remedy is independent binary)

### **3. Multi-Task Loss (Combined)**
- **Formula**: `Total Loss = 2.0 * FocalLoss + 1.0 * BCELoss`
- **Weighting**: Conventional treatment loss weighted 2x (more important)

---

## ⚖️ Class Imbalance Handling

### **1. Class Weights Calculation**
- **Algorithm**: `compute_class_weight('balanced')` from Scikit-learn
- **Formula**: `weight_i = n_samples / (n_classes * count_i)`
- **Applied to**: Both conventional treatment and natural remedies
- **Result**: Rare classes get higher weights (e.g., Ketoconazole weight = 2.46)

### **2. Positive Class Weights (for Multi-label)**
- **Algorithm**: Calculated per-label based on positive/negative ratio
- **Formula**: `pos_weight = (negative_samples / positive_samples)`
- **Purpose**: Balance positive vs negative examples for each natural remedy

---

## 🚀 Optimization Algorithms

### **1. Adam Optimizer**
- **Algorithm**: Adaptive Moment Estimation
- **Parameters**:
  - Learning Rate: `0.0005`
  - Weight Decay: `0.0001` (L2 regularization)
  - Beta1: `0.9` (default)
  - Beta2: `0.999` (default)
- **Why Used**: Adaptive learning rate, good for non-stationary objectives

### **2. Learning Rate Scheduling**
- **Algorithm**: `ReduceLROnPlateau`
- **Strategy**: Reduce learning rate by factor of 0.5 when validation loss plateaus
- **Patience**: 10 epochs
- **Min Learning Rate**: 1e-7
- **Purpose**: Fine-tune model when stuck in local minimum

---

## 🛡️ Regularization Techniques

### **1. Dropout**
- **Rate**: 0.4 (shared layers), 0.3 (conventional head), 0.2 (natural head)
- **Purpose**: Prevent overfitting by randomly zeroing neurons during training
- **Applied**: After each dense layer (except output layers)

### **2. Batch Normalization**
- **Algorithm**: Normalize activations across mini-batches
- **Formula**: `BN(x) = γ * (x - μ) / √(σ² + ε) + β`
- **Purpose**: Stabilize training, allow higher learning rates
- **Applied**: After each linear layer (before activation)

### **3. Weight Decay (L2 Regularization)**
- **Value**: `0.0001`
- **Purpose**: Penalize large weights to prevent overfitting
- **Applied**: Through Adam optimizer's `weight_decay` parameter

### **4. Label Smoothing**
- **Value**: `0.1`
- **Purpose**: Prevents model from being overconfident
- **Effect**: Softens hard labels (1.0 → 0.9, 0.0 → 0.1)

---

## 🎓 Training Strategies

### **1. Early Stopping**
- **Algorithm**: Monitor validation loss
- **Patience**: 25 epochs
- **Min Delta**: 0.0001
- **Purpose**: Stop training when model stops improving (prevents overfitting)

### **2. Mini-Batch Gradient Descent**
- **Batch Size**: 128 samples
- **Purpose**: Balance between gradient accuracy and training speed

### **3. Train/Validation/Test Split**
- **Split Ratio**: 70% / 15% / 15%
- **Random Seed**: 42 (for reproducibility)
- **Purpose**: Evaluate model on unseen data

---

## 📈 Evaluation Metrics

### **Conventional Treatment (Single-label Classification)**
1. **Accuracy**: Overall correctness
2. **F1-Score (Macro)**: Average F1 across all classes
3. **Precision (Macro)**: Average precision across classes
4. **Recall (Macro)**: Average recall across classes
5. **Confusion Matrix**: Per-class performance visualization

### **Natural Remedies (Multi-label Classification)**
1. **Hamming Loss**: Average fraction of labels incorrectly predicted
2. **F1-Score (Macro)**: Average F1 per label
3. **F1-Score (Micro)**: Global F1 across all labels
4. **Jaccard Score**: Intersection over union of predicted and true labels
5. **Precision/Recall (Macro)**: Average across all labels

---

## 🔧 Feature Engineering

### **1. Embedding Layers**
- **Algorithm**: Learnable dense vector representations
- **Dimension**: 64 per categorical feature
- **Purpose**: Capture semantic relationships between categories
- **Example**: Similar breeds get similar embeddings

### **2. Feature Concatenation**
- **Method**: Concatenate all embeddings + numerical features
- **Result**: Single feature vector for shared representation

---

## 🏗️ Model Architecture Details

### **Total Parameters**: ~789,519

**Breakdown**:
- Embedding layers: ~30,000 parameters
- Shared representation: ~500,000 parameters
- Conventional head: ~10,000 parameters
- Natural head: ~5,000 parameters

### **Activation Functions**
- **Shared Layers**: ReLU (Rectified Linear Unit)
- **Conventional Output**: Softmax (via CrossEntropyLoss)
- **Natural Output**: Sigmoid (for multi-label binary)

---

## 📚 Key Algorithms Summary

| Component | Algorithm/Technique | Purpose |
|-----------|-------------------|---------|
| **Architecture** | Deep Feedforward Neural Network | Learn complex patterns |
| **Categorical Encoding** | Embedding Layers | Learn semantic relationships |
| **Numerical Scaling** | StandardScaler (Z-score) | Normalize features |
| **Loss Function** | Focal Loss + BCE Loss | Handle class imbalance |
| **Optimizer** | Adam | Adaptive gradient descent |
| **Regularization** | Dropout + BatchNorm + Weight Decay | Prevent overfitting |
| **Class Balancing** | Class Weights + Positive Weights | Handle imbalanced data |
| **Learning Rate** | ReduceLROnPlateau | Adaptive learning rate |
| **Early Stopping** | Validation Loss Monitoring | Prevent overfitting |
| **Multi-Task Learning** | Shared Representation + Separate Heads | Learn both tasks simultaneously |

---

## 🔬 Advanced Techniques Used

1. **Multi-Task Learning**: Single model learns both conventional and natural treatments
2. **Transfer Learning**: Shared representation benefits both tasks
3. **Focal Loss**: Advanced loss function for imbalanced classification
4. **Label Smoothing**: Prevents overconfidence
5. **Class Weights**: Automatic balancing of imbalanced classes
6. **Batch Normalization**: Stabilizes training and allows higher learning rates

---

## 📖 References

1. **Focal Loss**: Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
2. **Adam Optimizer**: Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization." ICLR.
3. **Batch Normalization**: Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training." ICML.
4. **Multi-Task Learning**: Caruana, R. (1997). "Multitask Learning." Machine Learning.

---

**Status**: All algorithms implemented and tested ✅

