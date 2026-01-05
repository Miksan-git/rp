"""
Data Preprocessing Module for Dog Treatment Recommendation System

This module handles:
- Loading and cleaning the dataset
- Encoding categorical features (label encoding + embeddings)
- Normalizing numerical features
- Multi-label encoding for natural remedies
- Train/validation/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Tuple, Dict, List
import yaml


class DataPreprocessor:
    """
    Comprehensive data preprocessor for mixed categorical and numerical features
    with multi-label target encoding.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the preprocessor.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.mlb = MultiLabelBinarizer()
        self.feature_info = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_data(self, dataset_path: str) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Args:
            dataset_path: Path to the CSV dataset
            
        Returns:
            DataFrame with loaded data
        """
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset: handle missing values, normalize text, etc.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Handle missing values in categorical features
        categorical_cols = self.config.get('features', {}).get('categorical', [])
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('None')
        
        # Handle missing values in numerical features
        numerical_cols = self.config.get('features', {}).get('numerical', [])
        for col in numerical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Clean natural remedies (handle comma-separated values)
        natural_col = self.config.get('features', {}).get('targets', {}).get('natural_remedies', 'Natural Remedies')
        if natural_col in df_clean.columns:
            df_clean[natural_col] = df_clean[natural_col].apply(
                lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else []
            )
        
        print(f"Data cleaning completed. Remaining rows: {df_clean.shape[0]}")
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Encode categorical features using label encoding.
        Returns encoded indices for embedding lookup.
        
        Args:
            df: DataFrame with categorical features
            fit: Whether to fit new encoders (True for training, False for inference)
            
        Returns:
            Array of shape (n_samples, n_categorical_features) with encoded indices
        """
        categorical_cols = self.config.get('features', {}).get('categorical', [])
        encoded_features = []
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            if fit:
                le = LabelEncoder()
                encoded_values = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                self.feature_info[col] = {
                    'num_classes': len(le.classes_),
                    'classes': le.classes_.tolist()
                }
            else:
                le = self.label_encoders[col]
                # Handle unseen categories
                encoded_values = []
                for val in df[col].astype(str):
                    if val in le.classes_:
                        encoded_values.append(le.transform([val])[0])
                    else:
                        # Use most common class as default
                        encoded_values.append(0)
                encoded_values = np.array(encoded_values)
            
            encoded_features.append(encoded_values.reshape(-1, 1))
        
        return np.hstack(encoded_features) if encoded_features else np.array([]).reshape(len(df), 0)
    
    def encode_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Normalize numerical features using StandardScaler.
        
        Args:
            df: DataFrame with numerical features
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Array of shape (n_samples, n_numerical_features) with normalized values
        """
        numerical_cols = self.config.get('features', {}).get('numerical', [])
        
        if not numerical_cols:
            return np.array([]).reshape(len(df), 0)
        
        numerical_data = df[numerical_cols].values
        
        if fit:
            normalized_data = self.scaler.fit_transform(numerical_data)
        else:
            normalized_data = self.scaler.transform(numerical_data)
        
        return normalized_data
    
    def encode_conventional_treatment(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Encode conventional treatment (single-label classification).
        
        Args:
            df: DataFrame with target column
            fit: Whether to fit the encoder
            
        Returns:
            Array of shape (n_samples,) with encoded labels
        """
        target_col = self.config.get('features', {}).get('targets', {}).get('conventional_treatment', 'Conventional Treatment')
        
        if fit:
            le = LabelEncoder()
            encoded = le.fit_transform(df[target_col].astype(str))
            self.label_encoders['conventional_treatment'] = le
            self.feature_info['conventional_treatment'] = {
                'num_classes': len(le.classes_),
                'classes': le.classes_.tolist()
            }
        else:
            le = self.label_encoders['conventional_treatment']
            encoded = le.transform(df[target_col].astype(str))
        
        return encoded
    
    def encode_natural_remedies(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Encode natural remedies (multi-label classification).
        
        Args:
            df: DataFrame with target column
            fit: Whether to fit the multi-label binarizer
            
        Returns:
            Array of shape (n_samples, n_remedies) with binary labels
        """
        target_col = self.config.get('features', {}).get('targets', {}).get('natural_remedies', 'Natural Remedies')
        
        if target_col not in df.columns:
            return np.array([]).reshape(len(df), 0)
        
        # Extract list of remedies
        remedies_list = df[target_col].tolist()
        
        if fit:
            encoded = self.mlb.fit_transform(remedies_list)
            self.feature_info['natural_remedies'] = {
                'num_classes': len(self.mlb.classes_),
                'classes': self.mlb.classes_.tolist()
            }
        else:
            encoded = self.mlb.transform(remedies_list)
        
        return encoded
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare all features: categorical and numerical.
        
        Args:
            df: DataFrame with all features
            fit: Whether to fit encoders/scalers
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        cat_features = self.encode_categorical_features(df, fit=fit)
        num_features = self.encode_numerical_features(df, fit=fit)
        
        return cat_features, num_features
    
    def prepare_targets(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare all targets: conventional treatment and natural remedies.
        
        Args:
            df: DataFrame with target columns
            fit: Whether to fit encoders
            
        Returns:
            Tuple of (conventional_treatment, natural_remedies)
        """
        conventional = self.encode_conventional_treatment(df, fit=fit)
        natural = self.encode_natural_remedies(df, fit=fit)
        
        return conventional, natural
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Full DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_ratio = self.config.get('data', {}).get('train_split', 0.7)
        val_ratio = self.config.get('data', {}).get('val_split', 0.15)
        test_ratio = self.config.get('data', {}).get('test_split', 0.15)
        random_seed = self.config.get('data', {}).get('random_seed', 42)
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - train_ratio), 
            random_state=random_seed,
            shuffle=True
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=random_seed,
            shuffle=True
        )
        
        print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def save_preprocessor(self, save_path: str):
        """Save preprocessor state (encoders, scalers) to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        preprocessor_state = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'mlb': self.mlb,
            'feature_info': self.feature_info,
            'config': self.config
        }
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessor_state, f)
        print(f"Preprocessor saved to {save_path}")
    
    def load_preprocessor(self, load_path: str):
        """Load preprocessor state from disk."""
        with open(load_path, 'rb') as f:
            preprocessor_state = pickle.load(f)
        
        self.label_encoders = preprocessor_state['label_encoders']
        self.scaler = preprocessor_state['scaler']
        self.mlb = preprocessor_state['mlb']
        self.feature_info = preprocessor_state['feature_info']
        self.config = preprocessor_state.get('config', self.config)
        print(f"Preprocessor loaded from {load_path}")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(config_path="configs/config.yaml")
    df = preprocessor.load_data("Refined_Book_Aligned_Dog_Treatment_Dataset.csv")
    df_clean = preprocessor.clean_data(df)
    
    train_df, val_df, test_df = preprocessor.split_data(df_clean)
    
    # Prepare features and targets
    X_cat_train, X_num_train = preprocessor.prepare_features(train_df, fit=True)
    y_conv_train, y_nat_train = preprocessor.prepare_targets(train_df, fit=True)
    
    print(f"\nTraining set:")
    print(f"  Categorical features shape: {X_cat_train.shape}")
    print(f"  Numerical features shape: {X_num_train.shape}")
    print(f"  Conventional treatment shape: {y_conv_train.shape}")
    print(f"  Natural remedies shape: {y_nat_train.shape}")
    
    # Save preprocessor
    preprocessor.save_preprocessor("models/preprocessor.pkl")

