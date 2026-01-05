"""
Inference Script for Dog Treatment Recommendation System

This script loads a trained model and makes predictions on new data.
"""

import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import DataPreprocessor
from src.models.architecture import create_model_from_config


class TreatmentPredictor:
    """
    Predictor class for making treatment recommendations.
    """
    
    def __init__(self, model_path: str, preprocessor_path: str, config_path: str):
        """
        Initialize predictor with trained model and preprocessor.
        
        Args:
            model_path: Path to saved model checkpoint
            preprocessor_path: Path to saved preprocessor
            config_path: Path to configuration file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor(config_path=config_path)
        self.preprocessor.load_preprocessor(preprocessor_path)
        
        # Create model
        categorical_info = {}
        categorical_cols = self.config['features']['categorical']
        for col in categorical_cols:
            if col in self.preprocessor.feature_info:
                categorical_info[col] = self.preprocessor.feature_info[col]['num_classes']
        
        num_numerical = len(self.config['features']['numerical'])
        num_conv_classes = self.preprocessor.feature_info['conventional_treatment']['num_classes']
        num_nat_classes = self.preprocessor.feature_info['natural_remedies']['num_classes']
        
        self.model = create_model_from_config(
            categorical_info,
            num_numerical,
            num_conv_classes,
            num_nat_classes,
            self.config
        )
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model and preprocessor loaded successfully!")
    
    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on input data.
        
        Args:
            input_data: DataFrame with required features
        
        Returns:
            DataFrame with predictions
        """
        # Clean data
        input_clean = self.preprocessor.clean_data(input_data)
        
        # Prepare features
        X_cat, X_num = self.preprocessor.prepare_features(input_clean, fit=False)
        
        # Convert to tensors
        X_cat_tensor = torch.LongTensor(X_cat).to(self.device)
        X_num_tensor = torch.FloatTensor(X_num).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            conv_logits, nat_probs = self.model(X_cat_tensor, X_num_tensor)
            conv_preds = torch.argmax(conv_logits, dim=1)
            nat_preds = (nat_probs > 0.5).cpu().numpy()
        
        # Decode predictions
        conv_decoder = self.preprocessor.label_encoders['conventional_treatment']
        conv_classes = conv_decoder.inverse_transform(conv_preds.cpu().numpy())
        
        nat_decoder = self.preprocessor.mlb
        nat_classes = []
        for pred in nat_preds:
            indices = np.where(pred == 1)[0]
            remedies = [nat_decoder.classes_[i] for i in indices]
            nat_classes.append(', '.join(remedies) if remedies else 'None')
        
        # Create results DataFrame
        results = input_data.copy()
        results['Predicted_Conventional_Treatment'] = conv_classes
        results['Predicted_Natural_Remedies'] = nat_classes
        
        return results
    
    def predict_single(
        self,
        breed: str,
        age: float,
        weight: float,
        medical_history: str,
        genetic_predispositions: str,
        current_medications: str,
        diet: str,
        lifestyle: str,
        environment: str,
        vaccination_status: str,
        neutering_status: str,
        living_conditions: str,
        disease: str,
        stage: str
    ) -> dict:
        """
        Make prediction for a single dog case.
        
        Returns:
            Dictionary with predictions
        """
        input_dict = {
            'Breed': [breed],
            'Age': [age],
            'Weight': [weight],
            'Medical History': [medical_history],
            'Genetic Predispositions': [genetic_predispositions],
            'Current Medications': [current_medications],
            'Diet': [diet],
            'Lifestyle': [lifestyle],
            'Environment': [environment],
            'Vaccination Status': [vaccination_status],
            'Neutering Status': [neutering_status],
            'Living Conditions': [living_conditions],
            'Disease': [disease],
            'Stage': [stage]
        }
        
        input_df = pd.DataFrame(input_dict)
        results = self.predict(input_df)
        
        return {
            'conventional_treatment': results['Predicted_Conventional_Treatment'].iloc[0],
            'natural_remedies': results['Predicted_Natural_Remedies'].iloc[0]
        }


def main():
    """Example usage of the predictor."""
    model_path = "models/saved/best_model.pth"
    preprocessor_path = "models/saved/preprocessor.pkl"
    config_path = "configs/config.yaml"
    
    # Initialize predictor
    predictor = TreatmentPredictor(model_path, preprocessor_path, config_path)
    
    # Example: Predict for a single case
    result = predictor.predict_single(
        breed="Chihuahua",
        age=6,
        weight=2.8,
        medical_history="None",
        genetic_predispositions="None",
        current_medications="None",
        diet="Balanced",
        lifestyle="Indoor",
        environment="Temperate",
        vaccination_status="Not Up-to-date",
        neutering_status="Neutered",
        living_conditions="Multi-pet",
        disease="Hypersensitivity Allergic Dermatosis",
        stage="Severe"
    )
    
    print("\nPrediction Results:")
    print(f"Conventional Treatment: {result['conventional_treatment']}")
    print(f"Natural Remedies: {result['natural_remedies']}")


if __name__ == "__main__":
    main()

