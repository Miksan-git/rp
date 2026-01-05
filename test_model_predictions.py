"""
Test model predictions to see if it's making diverse predictions
"""
import torch
import pickle
import pandas as pd
import numpy as np
import yaml
import sys
import os
sys.path.append('.')

from src.data.preprocessing import DataPreprocessor
from src.models.architecture import create_model_from_config

# Load config
config_path = 'configs/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load preprocessor
preprocessor = DataPreprocessor(config_path=config_path)
preprocessor_path = 'models/saved/preprocessor.pkl'
preprocessor.load_preprocessor(preprocessor_path)

# Create model
categorical_info = {}
categorical_cols = config['features']['categorical']
for col in categorical_cols:
    if col in preprocessor.feature_info:
        categorical_info[col] = preprocessor.feature_info[col]['num_classes']

num_numerical = len(config['features']['numerical'])
num_conv_classes = preprocessor.feature_info['conventional_treatment']['num_classes']
num_nat_classes = preprocessor.feature_info['natural_remedies']['num_classes']

device = torch.device('cpu')
model = create_model_from_config(
    categorical_info, num_numerical, num_conv_classes, num_nat_classes, config
)

checkpoint = torch.load('models/saved/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

conv_decoder = preprocessor.label_encoders['conventional_treatment']
print("Conventional Treatment Classes:", conv_decoder.classes_)
print()

# Test cases
test_cases = [
    {
        'name': 'Bulldog - Hypersensitivity',
        'data': {
            'Breed': 'Bulldog', 'Age': 6, 'Weight': 22.0,
            'Medical History': 'Skin allergies', 'Genetic Predispositions': 'Dermatitis',
            'Current Medications': 'Insulin', 'Diet': 'Low-sodium', 'Lifestyle': 'Indoor',
            'Environment': 'Tropical', 'Vaccination Status': 'Up-to-date',
            'Neutering Status': 'Neutered', 'Living Conditions': 'Single-pet',
            'Disease': 'Hypersensitivity Allergic Dermatosis', 'Stage': 'Moderate'
        }
    },
    {
        'name': 'Labrador - Fungal',
        'data': {
            'Breed': 'Labrador', 'Age': 5, 'Weight': 25.0,
            'Medical History': 'None', 'Genetic Predispositions': 'Fungal susceptibility',
            'Current Medications': 'None', 'Diet': 'Grain-free', 'Lifestyle': 'Active',
            'Environment': 'Humid', 'Vaccination Status': 'Up-to-date',
            'Neutering Status': 'Neutered', 'Living Conditions': 'Multi-pet',
            'Disease': 'Fungal Infections', 'Stage': 'Initial'
        }
    },
    {
        'name': 'German Shepherd - Staph',
        'data': {
            'Breed': 'German Shepherd', 'Age': 8, 'Weight': 30.0,
            'Medical History': 'Eye infections', 'Genetic Predispositions': 'None',
            'Current Medications': 'Heartworm Preventative', 'Diet': 'High-protein',
            'Lifestyle': 'Outdoor', 'Environment': 'Cold',
            'Vaccination Status': 'Up-to-date', 'Neutering Status': 'Not Neutered',
            'Living Conditions': 'Multi-pet', 'Disease': 'Staph Infection', 'Stage': 'Severe'
        }
    },
    {
        'name': 'Chihuahua - Bacterial with Allergy',
        'data': {
            'Breed': 'Chihuahua', 'Age': 3, 'Weight': 2.5,
            'Medical History': 'Allergy to Penicillin', 'Genetic Predispositions': 'None',
            'Current Medications': 'Antihistamines', 'Diet': 'Hypoallergenic',
            'Lifestyle': 'Indoor', 'Environment': 'Temperate',
            'Vaccination Status': 'Not Up-to-date', 'Neutering Status': 'Neutered',
            'Living Conditions': 'Single-pet', 'Disease': 'Bacterial Dermatosis',
            'Stage': 'Mild', 'Drug Allergies': 'Penicillin'
        }
    }
]

print("="*80)
print("TESTING MODEL PREDICTIONS")
print("="*80)
print()

for test_case in test_cases:
    print(f"Test: {test_case['name']}")
    print(f"Disease: {test_case['data']['Disease']}, Stage: {test_case['data']['Stage']}")
    
    # Add optional features with defaults
    full_data = test_case['data'].copy()
    optional_features = {
        'Severity Score': 50.0,
        'Symptom Count': 5,
        'Treatments Tried Count': 0,
        'Previous Treatment': 'None',
        'Previous Treatment Response': 'None',
        'Drug Allergies': 'None',
        'Drug Interaction Risk': 'Low',
        'Cost Category': 'Moderate',
        'Availability Status': 'In Stock'
    }
    for key, default in optional_features.items():
        if key not in full_data:
            full_data[key] = default
    
    df = pd.DataFrame([full_data])
    input_clean = preprocessor.clean_data(df)
    X_cat, X_num = preprocessor.prepare_features(input_clean, fit=False)
    
    X_cat_tensor = torch.LongTensor(X_cat).to(device)
    X_num_tensor = torch.FloatTensor(X_num).to(device)
    
    with torch.no_grad():
        conv_logits, nat_probs = model(X_cat_tensor, X_num_tensor)
        conv_probs = torch.softmax(conv_logits, dim=1)
        conv_pred_idx = torch.argmax(conv_logits, dim=1).cpu().numpy()[0]
    
    predicted = conv_decoder.inverse_transform([conv_pred_idx])[0]
    print(f"Predicted: {predicted}")
    print("Probabilities:")
    for i, class_name in enumerate(conv_decoder.classes_):
        prob = conv_probs[0, i].item()
        marker = " <--" if i == conv_pred_idx else ""
        print(f"  {class_name}: {prob:.4f}{marker}")
    print()

