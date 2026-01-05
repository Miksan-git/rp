"""
PyTorch Dataset and DataLoader for Dog Treatment Recommendation System
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class DogTreatmentDataset(Dataset):
    """
    PyTorch Dataset for dog treatment recommendation.
    
    Handles:
    - Categorical features (for embedding lookup)
    - Numerical features (normalized)
    - Conventional treatment target (single-label)
    - Natural remedies target (multi-label)
    """
    
    def __init__(
        self,
        categorical_features: np.ndarray,
        numerical_features: np.ndarray,
        conventional_target: np.ndarray,
        natural_target: np.ndarray
    ):
        """
        Initialize dataset.
        
        Args:
            categorical_features: Array of shape (n_samples, n_categorical_features)
            numerical_features: Array of shape (n_samples, n_numerical_features)
            conventional_target: Array of shape (n_samples,) with class indices
            natural_target: Array of shape (n_samples, n_natural_classes) with binary labels
        """
        self.categorical_features = torch.LongTensor(categorical_features)
        self.numerical_features = torch.FloatTensor(numerical_features)
        self.conventional_target = torch.LongTensor(conventional_target)
        self.natural_target = torch.FloatTensor(natural_target)
        
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.categorical_features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single data sample.
        
        Returns:
            Tuple of (categorical_features, numerical_features, conventional_target, natural_target)
        """
        return (
            self.categorical_features[idx],
            self.numerical_features[idx],
            self.conventional_target[idx],
            self.natural_target[idx]
        )


def create_dataloaders(
    train_cat: np.ndarray,
    train_num: np.ndarray,
    train_conv: np.ndarray,
    train_nat: np.ndarray,
    val_cat: np.ndarray,
    val_num: np.ndarray,
    val_conv: np.ndarray,
    val_nat: np.ndarray,
    test_cat: np.ndarray = None,
    test_num: np.ndarray = None,
    test_conv: np.ndarray = None,
    test_nat: np.ndarray = None,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        train_cat, train_num, train_conv, train_nat: Training data
        val_cat, val_num, val_conv, val_nat: Validation data
        test_cat, test_num, test_conv, test_nat: Test data (optional)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = DogTreatmentDataset(
        train_cat, train_num, train_conv, train_nat
    )
    
    val_dataset = DogTreatmentDataset(
        val_cat, val_num, val_conv, val_nat
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Test loader (if provided)
    if test_cat is not None:
        test_dataset = DogTreatmentDataset(
            test_cat, test_num, test_conv, test_nat
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    else:
        test_loader = None
    
    return train_loader, val_loader, test_loader

