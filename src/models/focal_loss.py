"""
Focal Loss for handling class imbalance

Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

Where:
- α_t: class weight (handles class imbalance)
- γ: focusing parameter (down-weights easy examples)
- p_t: predicted probability for true class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for each class (tensor or None)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Logits from model (batch_size, num_classes)
            targets: True class indices (batch_size,)
        
        Returns:
            Focal loss value
        """
        num_classes = inputs.size(1)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Convert targets to one-hot
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            # Apply smoothing
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            # Compute cross entropy with smoothed labels
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(targets_one_hot * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

