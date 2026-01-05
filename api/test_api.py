"""
Test script for the Dog Treatment Recommendation API

Usage:
    python api/test_api.py
"""

import requests
import json

API_URL = "http://localhost:8080"

def test_health():
    """Test health check endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_single_prediction():
    """Test single prediction endpoint."""
    print("Testing /predict endpoint...")
    
    test_data = {
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
    
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("Testing /predict/batch endpoint...")
    
    test_data = {
        "profiles": [
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
            },
            {
                "breed": "Labrador",
                "age": 5,
                "weight": 25.0,
                "medical_history": "Skin allergies",
                "genetic_predispositions": "Dermatitis",
                "current_medications": "None",
                "diet": "Grain-free",
                "lifestyle": "Active",
                "environment": "Temperate",
                "vaccination_status": "Up-to-date",
                "neutering_status": "Neutered",
                "living_conditions": "Single-pet",
                "disease": "Fungal Infections",
                "stage": "Initial"
            }
        ]
    }
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("="*80)
    print("API Testing Script")
    print("="*80)
    print()
    
    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        
        print("="*80)
        print("All tests completed!")
        print("="*80)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server.")
        print("Make sure the server is running: python api/app.py")
    except Exception as e:
        print(f"ERROR: {str(e)}")

