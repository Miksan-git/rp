"""
Flask REST API for Dog Treatment Recommendation System

API Endpoints:
- POST /predict - Get treatment recommendations for a dog profile
- GET /health - Health check endpoint
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import sys
import os
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import DataPreprocessor
from src.models.architecture import create_model_from_config

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and preprocessor
model = None
preprocessor = None
device = None
config = None


def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor, device, config
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load preprocessor
    preprocessor_path = os.path.join(
        os.path.dirname(__file__), '..', 
        config['paths']['model_save_dir'], 
        'preprocessor.pkl'
    )
    preprocessor = DataPreprocessor(config_path=config_path)
    preprocessor.load_preprocessor(preprocessor_path)
    print("Preprocessor loaded successfully")
    
    # Create model
    categorical_info = {}
    categorical_cols = config['features']['categorical']
    for col in categorical_cols:
        if col in preprocessor.feature_info:
            categorical_info[col] = preprocessor.feature_info[col]['num_classes']
    
    num_numerical = len(config['features']['numerical'])
    num_conv_classes = preprocessor.feature_info['conventional_treatment']['num_classes']
    num_nat_classes = preprocessor.feature_info['natural_remedies']['num_classes']
    
    model = create_model_from_config(
        categorical_info,
        num_numerical,
        num_conv_classes,
        num_nat_classes,
        config
    )
    
    # Load model weights
    model_path = os.path.join(
        os.path.dirname(__file__), '..',
        config['paths']['model_save_dir'],
        'best_model.pth'
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not set'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict treatment recommendations for a dog profile.
    
    Expected JSON input:
    {
        "breed": "Chihuahua",
        "age": 6,
        "weight": 2.8,
        "medical_history": "None",
        "genetic_predispositions": "None",
        "current_medications": "None",
        "diet": "Balanced",
        "lifestyle": "Indoor",
        "environment": "Temperate",
        "vaccination_status": "Not Up-to-date",
        "neutering_status": "Neutered",
        "living_conditions": "Multi-pet",
        "disease": "Hypersensitivity Allergic Dermatosis",
        "stage": "Severe"
    }
    
    Returns:
    {
        "conventional_treatment": "Amoxicillin",
        "natural_remedies": ["Chamomile", "Aloe vera gel", "Probiotics"],
        "confidence": {
            "conventional": 0.85,
            "natural": [0.92, 0.88, 0.76, ...]
        }
    }
    """
    if model is None or preprocessor is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure model files exist.'
        }), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'breed', 'age', 'weight', 'medical_history', 'genetic_predispositions',
            'current_medications', 'diet', 'lifestyle', 'environment',
            'vaccination_status', 'neutering_status', 'living_conditions',
            'disease', 'stage'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Create DataFrame from input
        input_dict = {
            'Breed': [data['breed']],
            'Age': [float(data['age'])],
            'Weight': [float(data['weight'])],
            'Medical History': [data['medical_history']],
            'Genetic Predispositions': [data['genetic_predispositions']],
            'Current Medications': [data['current_medications']],
            'Diet': [data['diet']],
            'Lifestyle': [data['lifestyle']],
            'Environment': [data['environment']],
            'Vaccination Status': [data['vaccination_status']],
            'Neutering Status': [data['neutering_status']],
            'Living Conditions': [data['living_conditions']],
            'Disease': [data['disease']],
            'Stage': [data['stage']]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # Preprocess input
        input_clean = preprocessor.clean_data(input_df)
        X_cat, X_num = preprocessor.prepare_features(input_clean, fit=False)
        
        # Convert to tensors
        X_cat_tensor = torch.LongTensor(X_cat).to(device)
        X_num_tensor = torch.FloatTensor(X_num).to(device)
        
        # Make predictions
        with torch.no_grad():
            conv_logits, nat_probs = model(X_cat_tensor, X_num_tensor)
            
            # Get conventional treatment prediction
            conv_probs = torch.softmax(conv_logits, dim=1)
            conv_pred_idx = torch.argmax(conv_logits, dim=1).cpu().numpy()[0]
            conv_confidence = conv_probs[0, conv_pred_idx].item()
            
            # Decode conventional treatment
            conv_decoder = preprocessor.label_encoders['conventional_treatment']
            conv_prediction = conv_decoder.inverse_transform([conv_pred_idx])[0]
            
            # Get natural remedies predictions
            nat_probs_cpu = nat_probs.cpu().numpy()[0]
            nat_preds = (nat_probs_cpu > 0.5).astype(int)
            
            # Decode natural remedies
            nat_decoder = preprocessor.mlb
            nat_indices = np.where(nat_preds == 1)[0]
            nat_remedies = [nat_decoder.classes_[i] for i in nat_indices]
            nat_confidences = [float(nat_probs_cpu[i]) for i in nat_indices]
        
        # Prepare response
        response = {
            'conventional_treatment': conv_prediction,
            'conventional_confidence': round(conv_confidence, 4),
            'natural_remedies': nat_remedies,
            'natural_confidences': [round(conf, 4) for conf in nat_confidences],
            'all_treatment_probabilities': {
                conv_decoder.classes_[i]: round(conv_probs[0, i].item(), 4)
                for i in range(len(conv_decoder.classes_))
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict treatment recommendations for multiple dog profiles.
    
    Expected JSON input:
    {
        "profiles": [
            {
                "breed": "Chihuahua",
                "age": 6,
                ...
            },
            ...
        ]
    }
    """
    if model is None or preprocessor is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure model files exist.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'profiles' not in data:
            return jsonify({'error': 'No profiles array provided'}), 400
        
        profiles = data['profiles']
        results = []
        
        for profile in profiles:
            try:
                # Create DataFrame
                input_dict = {
                    'Breed': [profile['breed']],
                    'Age': [float(profile['age'])],
                    'Weight': [float(profile['weight'])],
                    'Medical History': [profile['medical_history']],
                    'Genetic Predispositions': [profile['genetic_predispositions']],
                    'Current Medications': [profile['current_medications']],
                    'Diet': [profile['diet']],
                    'Lifestyle': [profile['lifestyle']],
                    'Environment': [profile['environment']],
                    'Vaccination Status': [profile['vaccination_status']],
                    'Neutering Status': [profile['neutering_status']],
                    'Living Conditions': [profile['living_conditions']],
                    'Disease': [profile['disease']],
                    'Stage': [profile['stage']]
                }
                
                input_df = pd.DataFrame(input_dict)
                input_clean = preprocessor.clean_data(input_df)
                X_cat, X_num = preprocessor.prepare_features(input_clean, fit=False)
                
                X_cat_tensor = torch.LongTensor(X_cat).to(device)
                X_num_tensor = torch.FloatTensor(X_num).to(device)
                
                with torch.no_grad():
                    conv_logits, nat_probs = model(X_cat_tensor, X_num_tensor)
                    conv_probs = torch.softmax(conv_logits, dim=1)
                    conv_pred_idx = torch.argmax(conv_logits, dim=1).cpu().numpy()[0]
                    conv_confidence = conv_probs[0, conv_pred_idx].item()
                    
                    conv_decoder = preprocessor.label_encoders['conventional_treatment']
                    conv_prediction = conv_decoder.inverse_transform([conv_pred_idx])[0]
                    
                    nat_probs_cpu = nat_probs.cpu().numpy()[0]
                    nat_preds = (nat_probs_cpu > 0.5).astype(int)
                    nat_decoder = preprocessor.mlb
                    nat_indices = np.where(nat_preds == 1)[0]
                    nat_remedies = [nat_decoder.classes_[i] for i in nat_indices]
                
                results.append({
                    'conventional_treatment': conv_prediction,
                    'conventional_confidence': round(conv_confidence, 4),
                    'natural_remedies': nat_remedies
                })
            except Exception as e:
                results.append({
                    'error': f'Failed to process profile: {str(e)}'
                })
        
        return jsonify({'results': results}), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("\n" + "="*80)
    print("API Server Starting...")
    print("="*80)
    print("Endpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict - Single prediction")
    print("  POST /predict/batch - Batch predictions")
    print("\nServer running on http://localhost:8080")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=True)

