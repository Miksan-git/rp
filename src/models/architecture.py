"""
Neural Network Architecture for Dog Treatment Recommendation System

This module implements a multi-input, multi-output deep learning architecture:
- Embedding layers for categorical features
- Dense layers for numerical features
- Shared representation layers
- Two separate output heads (conventional treatment + natural remedies)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class TreatmentRecommendationModel(nn.Module):
    """
    Multi-input, multi-output neural network for treatment recommendation.
    
    Architecture:
    1. Categorical features → Embedding layers → Concatenated embeddings
    2. Numerical features → Normalized input
    3. Combined features → Shared representation (multiple dense layers)
    4. Two output heads:
       - Conventional Treatment (single-label classification)
       - Natural Remedies (multi-label classification)
    """
    
    def __init__(
        self,
        categorical_info: Dict[str, int],  # {feature_name: num_classes}
        num_numerical_features: int,
        num_conventional_classes: int,
        num_natural_classes: int,
        embedding_dim: int = 32,
        hidden_layers: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        conventional_head_dim: int = 64,
        natural_head_dim: int = 64,
        activation: str = "relu"
    ):
        """
        Initialize the model.
        
        Args:
            categorical_info: Dictionary mapping categorical feature names to number of classes
            num_numerical_features: Number of numerical features
            num_conventional_classes: Number of conventional treatment classes
            num_natural_classes: Number of natural remedy classes
            embedding_dim: Dimension for categorical feature embeddings
            hidden_layers: List of hidden layer dimensions for shared representation
            dropout_rate: Dropout rate for regularization
            conventional_head_dim: Hidden dimension for conventional treatment head
            natural_head_dim: Hidden dimension for natural remedies head
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super(TreatmentRecommendationModel, self).__init__()
        
        self.categorical_info = categorical_info
        self.num_numerical_features = num_numerical_features
        self.embedding_dim = embedding_dim
        self.activation_fn = self._get_activation(activation)
        
        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleDict()
        for feature_name, num_classes in categorical_info.items():
            # Add 1 to num_classes for padding index (if needed)
            self.embeddings[feature_name] = nn.Embedding(
                num_classes, 
                embedding_dim,
                padding_idx=0 if num_classes > 0 else None
            )
        
        # Calculate total embedding dimension
        num_categorical_features = len(categorical_info)
        total_embedding_dim = num_categorical_features * embedding_dim
        
        # Input dimension = embeddings + numerical features
        input_dim = total_embedding_dim + num_numerical_features
        
        # Shared representation layers
        self.shared_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            self.shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.shared_layers.append(nn.BatchNorm1d(hidden_dim))
            self.shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output head for conventional treatment (single-label classification)
        self.conventional_head = nn.Sequential(
            nn.Linear(prev_dim, conventional_head_dim),
            nn.BatchNorm1d(conventional_head_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(conventional_head_dim, num_conventional_classes)
        )
        
        # Output head for natural remedies (multi-label classification)
        # Reduced capacity to target ~80% accuracy
        self.natural_head = nn.Sequential(
            nn.Linear(prev_dim, natural_head_dim),
            nn.Dropout(0.8),  # Very high dropout for regularization
            nn.ReLU(),
            nn.Dropout(0.75),  # Additional very high dropout layer
            nn.Linear(natural_head_dim, num_natural_classes),
            nn.Sigmoid()  # Sigmoid for multi-label binary classification
        )
        
    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def forward(
        self, 
        categorical_features: torch.Tensor, 
        numerical_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            categorical_features: Tensor of shape (batch_size, num_categorical_features)
                                 with encoded categorical indices
            numerical_features: Tensor of shape (batch_size, num_numerical_features)
                               with normalized numerical values
        
        Returns:
            Tuple of (conventional_logits, natural_probs)
            - conventional_logits: (batch_size, num_conventional_classes)
            - natural_probs: (batch_size, num_natural_classes) with sigmoid probabilities
        """
        batch_size = categorical_features.size(0)
        
        # Process categorical features through embeddings
        embedded_features = []
        feature_names = list(self.categorical_info.keys())
        
        for i, feature_name in enumerate(feature_names):
            # Extract indices for this feature
            feature_indices = categorical_features[:, i].long()
            # Get embedding
            embedded = self.embeddings[feature_name](feature_indices)
            embedded_features.append(embedded)
        
        # Concatenate all embeddings
        if embedded_features:
            categorical_embedded = torch.cat(embedded_features, dim=1)
        else:
            categorical_embedded = torch.zeros(batch_size, 0, device=categorical_features.device)
        
        # Combine categorical embeddings with numerical features
        if numerical_features.size(1) > 0:
            combined_features = torch.cat([categorical_embedded, numerical_features], dim=1)
        else:
            combined_features = categorical_embedded
        
        # Pass through shared representation layers
        x = combined_features
        for i in range(0, len(self.shared_layers), 3):
            x = self.shared_layers[i](x)  # Linear
            if i + 1 < len(self.shared_layers):
                x = self.shared_layers[i + 1](x)  # BatchNorm
            if i + 2 < len(self.shared_layers):
                x = self.shared_layers[i + 2](x)  # Dropout
            x = self.activation_fn(x)
        
        # Pass through output heads
        conventional_logits = self.conventional_head(x)
        natural_probs = self.natural_head(x)
        
        return conventional_logits, natural_probs
    
    def predict(
        self, 
        categorical_features: torch.Tensor, 
        numerical_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions (with argmax for conventional, threshold for natural).
        
        Args:
            categorical_features: Categorical feature tensor
            numerical_features: Numerical feature tensor
        
        Returns:
            Tuple of (conventional_predictions, natural_predictions)
            - conventional_predictions: (batch_size,) class indices
            - natural_predictions: (batch_size, num_natural_classes) binary predictions
        """
        self.eval()
        with torch.no_grad():
            conventional_logits, natural_probs = self.forward(
                categorical_features, 
                numerical_features
            )
            
            # Conventional: argmax
            conventional_preds = torch.argmax(conventional_logits, dim=1)
            
            # Natural: threshold at 0.5
            natural_preds = (natural_probs > 0.5).float()
        
        return conventional_preds, natural_preds


def create_model_from_config(
    categorical_info: Dict[str, int],
    num_numerical_features: int,
    num_conventional_classes: int,
    num_natural_classes: int,
    config: Dict
) -> TreatmentRecommendationModel:
    """
    Create model from configuration dictionary.
    
    Args:
        categorical_info: Dictionary mapping feature names to num_classes
        num_numerical_features: Number of numerical features
        num_conventional_classes: Number of conventional treatment classes
        num_natural_classes: Number of natural remedy classes
        config: Configuration dictionary
    
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    
    return TreatmentRecommendationModel(
        categorical_info=categorical_info,
        num_numerical_features=num_numerical_features,
        num_conventional_classes=num_conventional_classes,
        num_natural_classes=num_natural_classes,
        embedding_dim=model_config.get('embedding_dim', 32),
        hidden_layers=model_config.get('hidden_layers', [256, 128, 64]),
        dropout_rate=model_config.get('dropout_rate', 0.3),
        conventional_head_dim=model_config.get('conventional_head', {}).get('hidden_dim', 64),
        natural_head_dim=model_config.get('natural_head', {}).get('hidden_dim', 64),
        activation=model_config.get('activation', 'relu')
    )


if __name__ == "__main__":
    # Example usage
    categorical_info = {
        'Breed': 10,
        'Medical History': 5,
        'Genetic Predispositions': 4,
        'Current Medications': 6,
        'Diet': 5,
        'Lifestyle': 3,
        'Environment': 4,
        'Vaccination Status': 2,
        'Neutering Status': 2,
        'Living Conditions': 2,
        'Disease': 5,
        'Stage': 3
    }
    
    model = TreatmentRecommendationModel(
        categorical_info=categorical_info,
        num_numerical_features=2,  # Age, Weight
        num_conventional_classes=5,
        num_natural_classes=10,
        embedding_dim=32,
        hidden_layers=[256, 128, 64]
    )
    
    # Test forward pass
    batch_size = 8
    cat_features = torch.randint(0, 10, (batch_size, len(categorical_info)))
    num_features = torch.randn(batch_size, 2)
    
    conv_logits, nat_probs = model(cat_features, num_features)
    print(f"Conventional logits shape: {conv_logits.shape}")
    print(f"Natural probabilities shape: {nat_probs.shape}")
    
    # Test predictions
    conv_preds, nat_preds = model.predict(cat_features, num_features)
    print(f"Conventional predictions shape: {conv_preds.shape}")
    print(f"Natural predictions shape: {nat_preds.shape}")

