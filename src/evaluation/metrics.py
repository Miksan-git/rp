"""
Evaluation Metrics for Treatment Recommendation System

Metrics for:
- Conventional Treatment: Accuracy, F1, Precision, Recall, Confusion Matrix
- Natural Remedies: Hamming Loss, F1 (macro/micro), Precision, Recall, Jaccard Score
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, hamming_loss, jaccard_score,
    classification_report
)
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns


class TreatmentMetrics:
    """
    Comprehensive evaluation metrics for both tasks.
    """
    
    def __init__(self, class_names_conv: List[str] = None, class_names_nat: List[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            class_names_conv: List of conventional treatment class names
            class_names_nat: List of natural remedy class names
        """
        self.class_names_conv = class_names_conv
        self.class_names_nat = class_names_nat
    
    def evaluate_conventional(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        Evaluate conventional treatment predictions (single-label).
        
        Args:
            y_true: True labels (n_samples,)
            y_pred: Predicted labels (n_samples,)
            average: Averaging strategy for multi-class metrics
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average=average, zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        return metrics
    
    def evaluate_natural(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate natural remedies predictions (multi-label).
        
        Args:
            y_true: True binary labels (n_samples, n_remedies)
            y_pred: Predicted binary labels (n_samples, n_remedies)
            threshold: Threshold for converting probabilities to binary (if y_pred is probabilities)
        
        Returns:
            Dictionary of metric names and values
        """
        # Ensure binary predictions
        if y_pred.max() <= 1.0 and y_pred.min() >= 0.0:
            # Might be probabilities, convert to binary
            y_pred_binary = (y_pred > threshold).astype(int)
        else:
            y_pred_binary = y_pred.astype(int)
        
        y_true_binary = y_true.astype(int)
        
        metrics = {
            'hamming_loss': hamming_loss(y_true_binary, y_pred_binary),
            'f1_macro': f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0),
            'precision_macro': precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0),
            'jaccard_score': jaccard_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
        }
        
        return metrics
    
    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        save_path: str = None
    ) -> np.ndarray:
        """
        Compute and optionally visualize confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names for visualization
            save_path: Path to save confusion matrix plot
        
        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if save_path:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm))
            )
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        return cm
    
    def print_classification_report(
        self,
        y_true_conv: np.ndarray,
        y_pred_conv: np.ndarray,
        y_true_nat: np.ndarray,
        y_pred_nat: np.ndarray,
        target_names_conv: List[str] = None,
        target_names_nat: List[str] = None
    ):
        """
        Print detailed classification reports.
        
        Args:
            y_true_conv, y_pred_conv: Conventional treatment labels
            y_true_nat, y_pred_nat: Natural remedies labels
            target_names_conv: Conventional treatment class names
            target_names_nat: Natural remedy class names
        """
        print("\n" + "="*80)
        print("CONVENTIONAL TREATMENT CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(
            y_true_conv,
            y_pred_conv,
            target_names=target_names_conv,
            zero_division=0
        ))
        
        print("\n" + "="*80)
        print("NATURAL REMEDIES CLASSIFICATION REPORT")
        print("="*80)
        # For multi-label, we need to print per-label metrics
        if target_names_nat:
            for i, name in enumerate(target_names_nat):
                print(f"\n{name}:")
                print(classification_report(
                    y_true_nat[:, i],
                    y_pred_nat[:, i],
                    target_names=['Not Recommended', 'Recommended'],
                    zero_division=0
                ))


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: TreatmentMetrics,
    class_names_conv: List[str] = None,
    class_names_nat: List[str] = None
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        metrics: TreatmentMetrics instance
        class_names_conv: Conventional treatment class names
        class_names_nat: Natural remedy class names
    
    Returns:
        Tuple of (conv_metrics, nat_metrics, y_true_conv, y_pred_conv, y_true_nat, y_pred_nat)
    """
    model.eval()
    
    all_conv_preds = []
    all_conv_targets = []
    all_nat_preds = []
    all_nat_targets = []
    
    with torch.no_grad():
        for cat_feat, num_feat, conv_target, nat_target in dataloader:
            cat_feat = cat_feat.to(device)
            num_feat = num_feat.to(device)
            
            conv_logits, nat_probs = model(cat_feat, num_feat)
            
            # Get predictions
            conv_preds = torch.argmax(conv_logits, dim=1)
            nat_preds = (nat_probs > 0.5).float()
            
            # Collect predictions
            all_conv_preds.append(conv_preds.cpu().numpy())
            all_conv_targets.append(conv_target.numpy())
            all_nat_preds.append(nat_preds.cpu().numpy())
            all_nat_targets.append(nat_target.numpy())
    
    # Concatenate all predictions
    y_pred_conv = np.concatenate(all_conv_preds, axis=0)
    y_true_conv = np.concatenate(all_conv_targets, axis=0)
    y_pred_nat = np.concatenate(all_nat_preds, axis=0)
    y_true_nat = np.concatenate(all_nat_targets, axis=0)
    
    # Compute metrics
    conv_metrics = metrics.evaluate_conventional(y_true_conv, y_pred_conv)
    nat_metrics = metrics.evaluate_natural(y_true_nat, y_pred_nat)
    
    return conv_metrics, nat_metrics, y_true_conv, y_pred_conv, y_true_nat, y_pred_nat

