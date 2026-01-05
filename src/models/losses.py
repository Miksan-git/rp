"""
Custom Loss Functions for Multi-Task Learning

Combines:
- FocalLoss or CrossEntropyLoss for conventional treatment (single-label)
- BCEWithLogitsLoss or BCELoss for natural remedies (multi-label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.models.focal_loss import FocalLoss


class MultiTaskLoss(nn.Module):
    """
    Combined loss function for multi-task learning.
    
    Loss = w1 * L_conventional + w2 * L_natural
    """
    
    def __init__(
        self,
        conventional_weight: float = 1.0,
        natural_weight: float = 1.0,
        use_class_weights: bool = False,
        conventional_class_weights: torch.Tensor = None,
        natural_class_weights: torch.Tensor = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0
    ):
        """
        Initialize multi-task loss.
        
        Args:
            conventional_weight: Weight for conventional treatment loss
            natural_weight: Weight for natural remedies loss
            use_class_weights: Whether to use class weights for imbalanced data
            conventional_class_weights: Class weights for conventional treatment
            natural_class_weights: Class weights for natural remedies
            use_focal_loss: Whether to use Focal Loss for conventional treatment
            focal_gamma: Gamma parameter for Focal Loss
        """
        super(MultiTaskLoss, self).__init__()
        
        self.conventional_weight = conventional_weight
        self.natural_weight = natural_weight
        
        # Loss functions - Use Focal Loss for better handling of class imbalance
        if use_focal_loss:
            if use_class_weights and conventional_class_weights is not None:
                self.conventional_loss = FocalLoss(
                    alpha=conventional_class_weights, 
                    gamma=focal_gamma,
                    label_smoothing=label_smoothing
                )
            else:
                self.conventional_loss = FocalLoss(
                    alpha=None, 
                    gamma=focal_gamma,
                    label_smoothing=label_smoothing
                )
        else:
            if use_class_weights and conventional_class_weights is not None:
                self.conventional_loss = nn.CrossEntropyLoss(weight=conventional_class_weights)
            else:
                self.conventional_loss = nn.CrossEntropyLoss()
        
        # For multi-label, we'll use pos_weight instead of weight
        if use_class_weights and natural_class_weights is not None:
            # BCELoss doesn't support per-label weights directly
            # We'll use BCEWithLogitsLoss with pos_weight for better handling
            self.natural_loss = nn.BCEWithLogitsLoss(pos_weight=natural_class_weights)
            self.use_logits_for_natural = True
        else:
            self.natural_loss = nn.BCELoss()
            self.use_logits_for_natural = False
    
    def forward(
        self,
        conventional_logits: torch.Tensor,
        natural_probs: torch.Tensor,
        conventional_target: torch.Tensor,
        natural_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            conventional_logits: Model output for conventional treatment (batch_size, n_classes)
            natural_probs: Model output for natural remedies (batch_size, n_remedies) - sigmoid already applied
            conventional_target: True labels for conventional treatment (batch_size,)
            natural_target: True labels for natural remedies (batch_size, n_remedies)
        
        Returns:
            Tuple of (total_loss, conventional_loss, natural_loss)
        """
        # Compute losses
        loss_conv = self.conventional_loss(conventional_logits, conventional_target)
        
        # For natural remedies: if using class weights, convert probs to logits
        if self.use_logits_for_natural:
            # Convert probabilities back to logits for BCEWithLogitsLoss
            eps = 1e-8
            natural_logits = torch.log(natural_probs + eps) - torch.log(1 - natural_probs + eps)
            loss_nat = self.natural_loss(natural_logits, natural_target)
        else:
            # BCELoss expects probabilities (sigmoid already applied)
            loss_nat = self.natural_loss(natural_probs, natural_target)
        
        # Combined loss
        total_loss = self.conventional_weight * loss_conv + self.natural_weight * loss_nat
        
        return total_loss, loss_conv, loss_nat

