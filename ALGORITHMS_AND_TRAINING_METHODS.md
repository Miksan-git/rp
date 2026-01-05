# Algorithms and Training Methods Used

## 🧠 **Core Machine Learning Algorithm**

### **Deep Neural Network (Multi-Task Learning)**
- **Architecture Type**: Multi-Input, Multi-Output Neural Network
- **Framework**: PyTorch
- **Model Type**: Feedforward Neural Network with Embeddings

---

## 📊 **Model Architecture**

### **1. Input Processing**
- **Categorical Features**: Embedding layers (18 categorical features)
  - Each categorical feature gets its own embedding layer
  - Embedding dimension: **64**
  - Total categorical features: 18

- **Numerical Features**: Direct input (5 numerical features)
  - Age, Weight, Severity Score, Symptom Count, Treatments Tried Count
  - Normalized using StandardScaler (mean=0, std=1)

### **2. Shared Representation Layers**
```
Input (Categorical Embeddings + Numerical) 
  → Dense Layer 1: 512 neurons + BatchNorm + Dropout(0.4) + ReLU
  → Dense Layer 2: 256 neurons + BatchNorm + Dropout(0.4) + ReLU
  → Dense Layer 3: 128 neurons + BatchNorm + Dropout(0.4) + ReLU
  → Dense Layer 4: 64 neurons + BatchNorm + Dropout(0.4) + ReLU
```

### **3. Output Heads**
- **Conventional Treatment Head** (Single-label classification):
  - Dense: 128 neurons + Dropout(0.3) + ReLU
  - Output: 5 classes (Amoxicillin, Cephalexin, Fluconazole, Itraconazole, Ketoconazole)
  - Activation: Softmax

- **Natural Remedies Head** (Multi-label classification):
  - Dense: 16 neurons + Dropout(0.6) + ReLU
  - Output: 7 classes (multiple remedies can be selected)
  - Activation: Sigmoid

**Total Parameters**: 785,105 trainable parameters

---

## 🎯 **Optimization Algorithm**

### **Adam Optimizer** (Adaptive Moment Estimation)
- **Algorithm**: Adam (Kingma & Ba, 2014)
- **Learning Rate**: 0.0005
- **Weight Decay**: 0.0001 (L2 regularization)
- **Beta1**: 0.9 (default)
- **Beta2**: 0.999 (default)
- **Epsilon**: 1e-8 (default)

**Why Adam?**
- Adaptive learning rates for each parameter
- Handles sparse gradients well
- Good for non-stationary objectives
- Combines benefits of AdaGrad and RMSProp

**Formula**:
```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
```

---

## 📉 **Loss Functions**

### **1. Multi-Task Loss Function**
Combines two losses with weighted sum:
```
Total Loss = w₁ * L_conventional + w₂ * L_natural
```
- `w₁` (conventional_weight): **5.0**
- `w₂` (natural_weight): **0.3**

### **2. Conventional Treatment Loss**

#### **Focal Loss** (Primary)
- **Algorithm**: Focal Loss (Lin et al., 2017)
- **Purpose**: Handles class imbalance by down-weighting easy examples
- **Gamma (γ)**: **5.0** (focusing parameter)
- **Alpha (α)**: Class weights (automatically calculated)
- **Label Smoothing**: **0.1**

**Formula**:
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```
Where:
- `p_t` = predicted probability for true class
- `γ` = focusing parameter (higher = more focus on hard examples)
- `α_t` = class weight for true class

**Why Focal Loss?**
- Your dataset has severe class imbalance (Cephalexin: 51%, Amoxicillin: 24%, etc.)
- Focal Loss automatically focuses on hard-to-classify examples
- Prevents model from always predicting majority class

#### **Alternative**: CrossEntropyLoss (if focal_loss disabled)
- Standard cross-entropy with optional class weights

### **3. Natural Remedies Loss**

#### **BCEWithLogitsLoss** (Binary Cross-Entropy with Logits)
- **Algorithm**: Binary Cross-Entropy for multi-label classification
- **Pos Weight**: Automatically calculated for each remedy (handles label imbalance)
- **Label Smoothing**: **0.12**

**Formula**:
```
BCE = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
```
Where:
- `y` = true label (0 or 1)
- `σ(x)` = sigmoid of logits
- Each remedy is predicted independently

**Why BCEWithLogitsLoss?**
- Natural remedies are multi-label (multiple remedies can be selected)
- Each remedy is a binary classification problem
- Pos_weight handles imbalance (some remedies are more common)

---

## 📈 **Learning Rate Scheduling**

### **ReduceLROnPlateau** (Reduce Learning Rate on Plateau)
- **Algorithm**: Adaptive learning rate reduction
- **Mode**: 'min' (monitor validation loss)
- **Factor**: 0.5 (reduce LR by 50%)
- **Patience**: 10 epochs
- **Min Learning Rate**: 1e-6

**How it works**:
- Monitors validation loss
- If validation loss doesn't improve for 10 epochs → reduce LR by 50%
- Continues until min_lr is reached
- Helps fine-tune model when stuck in local minima

**Example**:
```
Epoch 1-10: LR = 0.0005
Epoch 11: No improvement → LR = 0.00025
Epoch 21: No improvement → LR = 0.000125
...
```

---

## 🛑 **Early Stopping**

### **Early Stopping Algorithm**
- **Patience**: 25 epochs
- **Min Delta**: 0.0001 (minimum improvement required)
- **Mode**: 'min' (stop when validation loss stops decreasing)

**How it works**:
- Monitors validation loss each epoch
- If validation loss doesn't improve by at least 0.0001 for 25 epochs → stop training
- Prevents overfitting
- Saves best model (lowest validation loss)

**Algorithm**:
```
if val_loss < best_val_loss - min_delta:
    best_val_loss = val_loss
    counter = 0
    save_model()
else:
    counter += 1
    if counter >= patience:
        stop_training()
```

---

## ⚖️ **Class Imbalance Handling**

### **1. Class Weights for Conventional Treatment**
- **Algorithm**: sklearn's `compute_class_weight('balanced')`
- **Formula**: `weight[i] = n_samples / (n_classes * np.bincount(y)[i])`

**Calculated Weights** (example):
- Amoxicillin: 0.83
- Cephalexin: 0.39 (most common, lowest weight)
- Fluconazole: 3.30 (rare, highest weight)
- Itraconazole: 1.50
- Ketoconazole: 3.73

**Purpose**: Give more importance to rare classes during training

### **2. Pos Weights for Natural Remedies**
- **Algorithm**: `pos_weight = negative_count / positive_count`
- Calculated separately for each remedy
- Handles imbalance in multi-label setting

---

## 🎲 **Regularization Techniques**

### **1. Dropout**
- **Shared Layers**: 0.4 (40% neurons randomly set to 0)
- **Conventional Head**: 0.3 (30% dropout)
- **Natural Head**: 0.6 (60% dropout - higher to reduce overfitting)

**Purpose**: Prevents overfitting by randomly disabling neurons

### **2. Weight Decay (L2 Regularization)**
- **Value**: 0.0001
- **Formula**: `loss = original_loss + λ * Σ(θ²)`
- **Purpose**: Penalizes large weights, prevents overfitting

### **3. Batch Normalization**
- Applied after each dense layer
- Normalizes activations: `(x - μ) / √(σ² + ε)`
- **Purpose**: Stabilizes training, allows higher learning rates

### **4. Label Smoothing**
- **Conventional**: 0.1 (10% smoothing)
- **Natural**: 0.12 (12% smoothing)
- **Formula**: `y_smooth = y * (1 - α) + α / num_classes`
- **Purpose**: Prevents overconfident predictions, improves generalization

---

## 📦 **Data Preprocessing**

### **1. Categorical Encoding**
- **Algorithm**: Label Encoding
- Each categorical value → integer index
- Then passed through embedding layer

### **2. Numerical Normalization**
- **Algorithm**: StandardScaler (Z-score normalization)
- **Formula**: `x_normalized = (x - μ) / σ`
- **Purpose**: All features on same scale (mean=0, std=1)

### **3. Multi-Label Encoding**
- **Algorithm**: MultiLabelBinarizer
- Converts comma-separated remedies → binary vector
- Example: "Aloe Vera, Coconut Oil" → [1, 0, 1, 0, 0, 0, 0]

---

## 🔄 **Training Process**

### **Training Loop Algorithm**:
```
1. Initialize model, optimizer, loss function
2. For each epoch:
   a. Set model to training mode
   b. For each batch:
      - Forward pass: model(inputs) → predictions
      - Compute loss: criterion(predictions, targets)
      - Backward pass: loss.backward() (compute gradients)
      - Update weights: optimizer.step()
   c. Validate on validation set
   d. Update learning rate (if needed)
   e. Check early stopping
   f. Save best model
3. Load best model
4. Evaluate on test set
```

### **Gradient Descent**:
- **Type**: Mini-batch Gradient Descent
- **Batch Size**: 128 samples per batch
- **Gradient Computation**: Automatic differentiation (PyTorch autograd)
- **Weight Update**: Adam optimizer

---

## 📊 **Evaluation Metrics**

### **Conventional Treatment** (Single-label):
- **Accuracy**: Overall correctness
- **F1-Score (Macro)**: Average F1 across all classes
- **Precision (Macro)**: Average precision
- **Recall (Macro)**: Average recall
- **Confusion Matrix**: Per-class performance

### **Natural Remedies** (Multi-label):
- **Hamming Loss**: Fraction of incorrectly predicted labels
- **F1-Score (Macro/Micro)**: Multi-label F1 scores
- **Precision/Recall (Macro)**: Average across labels
- **Jaccard Score**: Intersection over union

---

## 🎯 **Key Algorithmic Choices Summary**

| Component | Algorithm | Reason |
|-----------|-----------|--------|
| **Optimizer** | Adam | Adaptive learning rates, handles sparse gradients |
| **Conventional Loss** | Focal Loss (γ=5.0) | Handles severe class imbalance |
| **Natural Loss** | BCEWithLogitsLoss | Multi-label binary classification |
| **LR Scheduler** | ReduceLROnPlateau | Fine-tune when stuck |
| **Regularization** | Dropout + Weight Decay + BatchNorm | Prevent overfitting |
| **Class Imbalance** | Class Weights + Focal Loss | Handle imbalanced dataset |
| **Early Stopping** | Validation loss monitoring | Prevent overfitting |

---

## 📚 **References**

1. **Adam Optimizer**: Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization"
2. **Focal Loss**: Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection"
3. **Batch Normalization**: Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training"
4. **Dropout**: Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"

---

## 🔬 **Training Configuration**

- **Epochs**: Up to 100 (with early stopping)
- **Batch Size**: 128
- **Train/Val/Test Split**: 70% / 15% / 15%
- **Random Seed**: 42 (for reproducibility)
- **Device**: CPU or CUDA (GPU if available)

---

## 💡 **Why These Algorithms?**

1. **Multi-Task Learning**: Predicts both conventional and natural treatments simultaneously, sharing learned features
2. **Focal Loss**: Essential for your imbalanced dataset (51% Cephalexin vs 5% Ketoconazole)
3. **Adam**: Industry standard, works well for most deep learning tasks
4. **ReduceLROnPlateau**: Helps fine-tune when model plateaus
5. **Early Stopping**: Prevents overfitting, saves training time
6. **Class Weights**: Gives rare classes more importance during training

This combination of algorithms is specifically chosen to handle:
- ✅ Class imbalance
- ✅ Multi-task learning
- ✅ Overfitting prevention
- ✅ Efficient training

