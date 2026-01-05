"""
Enhanced Training Script - Handles new features from enhanced dataset
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
import pandas as pd
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


def detect_new_features(df: pd.DataFrame) -> dict:
    """Detect new features in enhanced dataset."""
    original_features = [
        'Breed', 'Medical History', 'Genetic Predispositions',
        'Current Medications', 'Diet', 'Lifestyle', 'Environment',
        'Vaccination Status', 'Neutering Status', 'Living Conditions',
        'Disease', 'Stage'
    ]
    
    all_cols = df.columns.tolist()
    exclude_cols = ['Case ID', 'Age', 'Weight', 'Conventional Treatment', 
                    'Natural Remedies', 'Outcome']
    
    categorical_new = []
    numerical_new = []
    
    for col in all_cols:
        if col in exclude_cols:
            continue
        if col not in original_features:
            # Determine if categorical or numerical
            if df[col].dtype in ['object', 'string', 'category']:
                categorical_new.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                numerical_new.append(col)
    
    return {
        'categorical': categorical_new,
        'numerical': numerical_new
    }


def update_config_with_new_features(config_path: str, new_features: dict):
    """Update config file with new features."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add new categorical features
    if new_features['categorical']:
        config['features']['categorical'].extend(new_features['categorical'])
        print(f"\n✅ Added {len(new_features['categorical'])} new categorical features:")
        for feat in new_features['categorical']:
            print(f"   - {feat}")
    
    # Add new numerical features
    if new_features['numerical']:
        config['features']['numerical'].extend(new_features['numerical'])
        print(f"\n✅ Added {len(new_features['numerical'])} new numerical features:")
        for feat in new_features['numerical']:
            print(f"   - {feat}")
    
    # Update dataset path
    config['data']['dataset_path'] = 'Enhanced_Dog_Treatment_Dataset.csv'
    
    # Save updated config
    backup_path = config_path.replace('.yaml', '_backup.yaml')
    import shutil
    shutil.copy(config_path, backup_path)
    print(f"\n✅ Backed up original config to: {backup_path}")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Updated config saved to: {config_path}")
    
    return config


def main():
    """Main training function."""
    print("="*80)
    print("ENHANCED DATASET TRAINING")
    print("="*80)
    
    # Load config first to get dataset path
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    enhanced_file = config['data']['dataset_path']
    if not os.path.exists(enhanced_file):
        print(f"\n❌ Dataset not found: {enhanced_file}")
        print(f"\nAvailable datasets:")
        import glob
        for f in glob.glob("*.csv"):
            print(f"  - {f}")
        return
    
    # Load config
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and analyze dataset
    print(f"\n{'='*80}")
    print("LOADING DATASET")
    print("="*80)
    
    dataset_path = config['data']['dataset_path']
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"✅ Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Detect new features and update config if needed
    new_features = detect_new_features(df)
    if new_features['categorical'] or new_features['numerical']:
        print(f"\n📊 New features detected:")
        print(f"   Categorical: {len(new_features['categorical'])}")
        print(f"   Numerical: {len(new_features['numerical'])}")
        config = update_config_with_new_features(config_path, new_features)
    else:
        print("\n✅ Using existing config features")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
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
    df_loaded = preprocessor.load_data(config['data']['dataset_path'])
    df_clean = preprocessor.clean_data(df_loaded)
    
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
        os.path.join(config['paths']['model_save_dir'], 'preprocessor_enhanced.pkl')
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
    
    # Calculate class weights
    use_class_weights = config['training'].get('use_class_weights', False)
    conventional_class_weights = None
    natural_class_weights = None
    
    if use_class_weights:
        print("\nCalculating class weights for imbalanced data...")
        conv_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_conv_train),
            y=y_conv_train
        )
        conventional_class_weights = torch.FloatTensor(conv_weights).to(device)
        print(f"Conventional treatment class weights: {conventional_class_weights.cpu().numpy()}")
        
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
        label_smoothing=config['training'].get('label_smoothing', 0.0)
    )
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
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
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_enhanced"
    )
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_val_loss = float('inf')
    best_accuracy = 0.0
    
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
            }, os.path.join(config['paths']['model_save_dir'], 'best_model_enhanced.pth'))
            print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    writer.close()
    
    # Load best model for evaluation
    print("\n" + "="*80)
    print("EVALUATING BEST MODEL")
    print("="*80)
    
    checkpoint = torch.load(os.path.join(config['paths']['model_save_dir'], 'best_model_enhanced.pth'))
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
    
    print("\n" + "="*80)
    print("FINAL RESULTS - ENHANCED MODEL")
    print("="*80)
    
    print("\nConventional Treatment Metrics:")
    for metric, value in conv_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nNatural Remedies Metrics:")
    for metric, value in nat_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Compare with previous accuracy
    print("\n" + "="*80)
    print("ACCURACY COMPARISON")
    print("="*80)
    print(f"Previous Model Accuracy: ~45-50%")
    print(f"Enhanced Model Accuracy: {conv_metrics['accuracy']*100:.2f}%")
    improvement = (conv_metrics['accuracy'] - 0.475) * 100
    print(f"Improvement: {improvement:+.2f} percentage points")
    
    if conv_metrics['accuracy'] > 0.70:
        print("\n🎉 EXCELLENT! Accuracy improved significantly!")
    elif conv_metrics['accuracy'] > 0.60:
        print("\n✅ GOOD! Accuracy improved!")
    else:
        print("\n⚠️  Accuracy still needs improvement. Check feature quality.")
    
    # Print classification reports
    metrics.print_classification_report(
        y_true_conv, y_pred_conv, y_true_nat, y_pred_nat,
        target_names_conv=preprocessor.feature_info['conventional_treatment']['classes'],
        target_names_nat=preprocessor.feature_info['natural_remedies']['classes']
    )
    
    # Save confusion matrix
    cm_path = os.path.join(config['paths']['results_dir'], 'confusion_matrix_enhanced.png')
    metrics.compute_confusion_matrix(
        y_true_conv, y_pred_conv,
        class_names=preprocessor.feature_info['conventional_treatment']['classes'],
        save_path=cm_path
    )
    
    print(f"\n✅ Training completed!")
    print(f"   Model saved to: {config['paths']['model_save_dir']}/best_model_enhanced.pth")
    print(f"   Results saved to: {config['paths']['results_dir']}")


if __name__ == "__main__":
    main()

