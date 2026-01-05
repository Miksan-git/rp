"""
Training Script for Dog Treatment Recommendation System

This script:
1. Loads and preprocesses data
2. Creates model architecture
3. Trains the model with early stopping
4. Evaluates on validation and test sets
5. Saves model and preprocessor
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
from typing import Tuple
from sklearn.utils.class_weight import compute_class_weight

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import DataPreprocessor
from src.data.dataset import create_dataloaders
from src.models.architecture import create_model_from_config
from src.models.losses import MultiTaskLoss
from src.evaluation.metrics import TreatmentMetrics, evaluate_model


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_conv_loss = 0.0
    total_nat_loss = 0.0
    num_batches = 0
    
    for cat_feat, num_feat, conv_target, nat_target in tqdm(dataloader, desc="Training"):
        cat_feat = cat_feat.to(device)
        num_feat = num_feat.to(device)
        conv_target = conv_target.to(device)
        nat_target = nat_target.to(device)
        
        # Forward pass
        conv_logits, nat_probs = model(cat_feat, num_feat)
        
        # Compute loss
        loss, conv_loss, nat_loss = criterion(
            conv_logits, nat_probs, conv_target, nat_target
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_conv_loss += conv_loss.item()
        total_nat_loss += nat_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_conv_loss = total_conv_loss / num_batches
    avg_nat_loss = total_nat_loss / num_batches
    
    return avg_loss, avg_conv_loss, avg_nat_loss


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device
) -> Tuple[float, float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_conv_loss = 0.0
    total_nat_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for cat_feat, num_feat, conv_target, nat_target in dataloader:
            cat_feat = cat_feat.to(device)
            num_feat = num_feat.to(device)
            conv_target = conv_target.to(device)
            nat_target = nat_target.to(device)
            
            conv_logits, nat_probs = model(cat_feat, num_feat)
            
            loss, conv_loss, nat_loss = criterion(
                conv_logits, nat_probs, conv_target, nat_target
            )
            
            total_loss += loss.item()
            total_conv_loss += conv_loss.item()
            total_nat_loss += nat_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_conv_loss = total_conv_loss / num_batches
    avg_nat_loss = total_nat_loss / num_batches
    
    return avg_loss, avg_conv_loss, avg_nat_loss


def main():
    """Main training function."""
    # Load configuration
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['paths']['model_save_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    # Initialize preprocessor
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    preprocessor = DataPreprocessor(config_path=config_path)
    
    # Load and clean data
    df = preprocessor.load_data(config['data']['dataset_path'])
    df_clean = preprocessor.clean_data(df)
    
    # Split data
    train_df, val_df, test_df = preprocessor.split_data(df_clean)
    
    # Prepare features and targets
    X_cat_train, X_num_train = preprocessor.prepare_features(train_df, fit=True)
    y_conv_train, y_nat_train = preprocessor.prepare_targets(train_df, fit=True)
    
    X_cat_val, X_num_val = preprocessor.prepare_features(val_df, fit=False)
    y_conv_val, y_nat_val = preprocessor.prepare_targets(val_df, fit=False)
    
    X_cat_test, X_num_test = preprocessor.prepare_features(test_df, fit=False)
    y_conv_test, y_nat_test = preprocessor.prepare_targets(test_df, fit=False)
    
    print(f"\nFeature shapes:")
    print(f"  Categorical: {X_cat_train.shape}")
    print(f"  Numerical: {X_num_train.shape}")
    print(f"  Conventional target: {y_conv_train.shape}")
    print(f"  Natural target: {y_nat_train.shape}")
    
    # Save preprocessor
    preprocessor.save_preprocessor(
        os.path.join(config['paths']['model_save_dir'], 'preprocessor.pkl')
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_cat_train, X_num_train, y_conv_train, y_nat_train,
        X_cat_val, X_num_val, y_conv_val, y_nat_val,
        X_cat_test, X_num_test, y_conv_test, y_nat_test,
        batch_size=config['training']['batch_size']
    )
    
    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    # Get categorical feature info
    categorical_info = {}
    categorical_cols = config['features']['categorical']
    for col in categorical_cols:
        if col in preprocessor.feature_info:
            categorical_info[col] = preprocessor.feature_info[col]['num_classes']
    
    num_numerical = X_num_train.shape[1]
    num_conv_classes = preprocessor.feature_info['conventional_treatment']['num_classes']
    num_nat_classes = preprocessor.feature_info['natural_remedies']['num_classes']
    
    model = create_model_from_config(
        categorical_info,
        num_numerical,
        num_conv_classes,
        num_nat_classes,
        config
    )
    
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Calculate class weights for imbalanced data
    use_class_weights = config['training'].get('use_class_weights', False)
    conventional_class_weights = None
    natural_class_weights = None
    
    if use_class_weights:
        print("\nCalculating class weights for imbalanced data...")
        # Calculate weights for conventional treatment
        conv_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_conv_train),
            y=y_conv_train
        )
        conventional_class_weights = torch.FloatTensor(conv_weights).to(device)
        print(f"Conventional treatment class weights: {conventional_class_weights.cpu().numpy()}")
        
        # For natural remedies, calculate pos_weight (ratio of negative to positive)
        # pos_weight = (num_negative / num_positive) for each label
        nat_pos_weights = []
        for i in range(y_nat_train.shape[1]):
            positive_count = np.sum(y_nat_train[:, i] == 1)
            negative_count = len(y_nat_train) - positive_count
            if positive_count > 0:
                pos_weight = negative_count / positive_count
            else:
                pos_weight = 1.0
            nat_pos_weights.append(pos_weight)
        natural_class_weights = torch.FloatTensor(nat_pos_weights).to(device)
        print(f"Natural remedies pos_weights calculated (avg: {natural_class_weights.mean().item():.2f})")
    
    # Loss function
    criterion = MultiTaskLoss(
        conventional_weight=config['training']['conventional_loss_weight'],
        natural_weight=config['training']['natural_loss_weight'],
        use_class_weights=use_class_weights,
        conventional_class_weights=conventional_class_weights,
        natural_class_weights=natural_class_weights,
        use_focal_loss=config['training'].get('use_focal_loss', True),
        focal_gamma=config['training'].get('focal_gamma', 2.0),
        label_smoothing=config['training'].get('label_smoothing', 0.0),
        natural_label_smoothing=config['training'].get('natural_label_smoothing', 0.0)
    )
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler - more patience to allow more training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        min_delta=config['training']['early_stopping_min_delta']
    )
    
    # TensorBoard writer
    log_dir = os.path.join(
        config['paths']['logs_dir'],
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_conv_loss, train_nat_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_conv_loss, val_nat_loss = validate(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Loss/Train_Conv', train_conv_loss, epoch)
        writer.add_scalar('Loss/Train_Nat', train_nat_loss, epoch)
        writer.add_scalar('Loss/Val_Conv', val_conv_loss, epoch)
        writer.add_scalar('Loss/Val_Nat', val_nat_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f} (Conv: {train_conv_loss:.4f}, Nat: {train_nat_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Conv: {val_conv_loss:.4f}, Nat: {val_nat_loss:.4f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(config['paths']['model_save_dir'], 'best_model.pth'))
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    writer.close()
    
    # Load best model for evaluation
    print("\n" + "="*80)
    print("EVALUATING BEST MODEL")
    print("="*80)
    
    checkpoint = torch.load(os.path.join(config['paths']['model_save_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize metrics
    metrics = TreatmentMetrics(
        class_names_conv=preprocessor.feature_info['conventional_treatment']['classes'],
        class_names_nat=preprocessor.feature_info['natural_remedies']['classes']
    )
    
    # Evaluate on test set
    conv_metrics, nat_metrics, y_true_conv, y_pred_conv, y_true_nat, y_pred_nat = evaluate_model(
        model, test_loader, device, metrics,
        class_names_conv=preprocessor.feature_info['conventional_treatment']['classes'],
        class_names_nat=preprocessor.feature_info['natural_remedies']['classes']
    )
    
    print("\nTest Set Results:")
    print("\nConventional Treatment Metrics:")
    for metric, value in conv_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nNatural Remedies Metrics:")
    for metric, value in nat_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Print classification reports
    metrics.print_classification_report(
        y_true_conv, y_pred_conv, y_true_nat, y_pred_nat,
        target_names_conv=preprocessor.feature_info['conventional_treatment']['classes'],
        target_names_nat=preprocessor.feature_info['natural_remedies']['classes']
    )
    
    # Save confusion matrix
    cm_path = os.path.join(config['paths']['results_dir'], 'confusion_matrix.png')
    metrics.compute_confusion_matrix(
        y_true_conv, y_pred_conv,
        class_names=preprocessor.feature_info['conventional_treatment']['classes'],
        save_path=cm_path
    )
    
    print(f"\nTraining completed! Model saved to {config['paths']['model_save_dir']}")
    print(f"Results saved to {config['paths']['results_dir']}")


if __name__ == "__main__":
    main()

