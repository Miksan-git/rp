"""
Test API with different scenarios
Run this after starting the API server: python3 api/app.py
"""

import requests
import json
import time

API_URL = "http://localhost:8080"

def test_request(name, data):
    """Test a single request."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print('='*80)
    print(f"Request: {json.dumps(data, indent=2)}")
    print("\nResponse:")
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            print(f"\n✅ Conventional Treatment: {result.get('conventional_treatment')} (Confidence: {result.get('conventional_confidence', 0):.2%})")
            print(f"✅ Natural Remedies: {', '.join(result.get('natural_remedies', []))}")
        else:
            print(f"❌ Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to API server.")
        print("   Make sure the server is running: python3 api/app.py")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
    
    time.sleep(0.5)  # Small delay between requests

# Test scenarios
test_cases = [
    {
        "name": "1. Bulldog - Hypersensitivity (Your Request)",
        "data": {
            "breed": "Bulldog",
            "age": 6,
            "weight": 22.0,
            "medical_history": "Skin allergies",
            "genetic_predispositions": "Dermatitis",
            "current_medications": "Insulin",
            "diet": "Low-sodium",
            "lifestyle": "Indoor",
            "environment": "Tropical",
            "vaccination_status": "Up-to-date",
            "neutering_status": "Neutered",
            "living_conditions": "Single-pet",
            "disease": "Hypersensitivity Allergic Dermatosis",
            "stage": "Moderate"
        }
    },
    {
        "name": "2. Labrador - Fungal Infection (Initial)",
        "data": {
            "breed": "Labrador",
            "age": 5,
            "weight": 25.0,
            "medical_history": "None",
            "genetic_predispositions": "Fungal susceptibility",
            "current_medications": "None",
            "diet": "Grain-free",
            "lifestyle": "Active",
            "environment": "Humid",
            "vaccination_status": "Up-to-date",
            "neutering_status": "Neutered",
            "living_conditions": "Multi-pet",
            "disease": "Fungal Infections",
            "stage": "Initial"
        }
    },
    {
        "name": "3. German Shepherd - Staph Infection (Severe)",
        "data": {
            "breed": "German Shepherd",
            "age": 8,
            "weight": 30.0,
            "medical_history": "Eye infections",
            "genetic_predispositions": "None",
            "current_medications": "Heartworm Preventative",
            "diet": "High-protein",
            "lifestyle": "Outdoor",
            "environment": "Cold",
            "vaccination_status": "Up-to-date",
            "neutering_status": "Not Neutered",
            "living_conditions": "Multi-pet",
            "disease": "Staph Infection",
            "stage": "Severe"
        }
    },
    {
        "name": "4. Chihuahua - Bacterial with Penicillin Allergy",
        "data": {
            "breed": "Chihuahua",
            "age": 3,
            "weight": 2.5,
            "medical_history": "Allergy to Penicillin",
            "genetic_predispositions": "None",
            "current_medications": "Antihistamines",
            "diet": "Hypoallergenic",
            "lifestyle": "Indoor",
            "environment": "Temperate",
            "vaccination_status": "Not Up-to-date",
            "neutering_status": "Neutered",
            "living_conditions": "Single-pet",
            "disease": "Bacterial Dermatosis",
            "stage": "Mild",
            "severity_score": 35,
            "symptom_count": 2,
            "drug_allergies": "Penicillin"
        }
    },
    {
        "name": "5. Poodle - Previous Treatment Failed",
        "data": {
            "breed": "Poodle",
            "age": 7,
            "weight": 15.0,
            "medical_history": "Skin allergies",
            "genetic_predispositions": "Dermatitis",
            "current_medications": "None",
            "diet": "Balanced",
            "lifestyle": "Indoor",
            "environment": "Temperate",
            "vaccination_status": "Up-to-date",
            "neutering_status": "Neutered",
            "living_conditions": "Single-pet",
            "disease": "Bacterial Dermatosis",
            "stage": "Moderate",
            "previous_treatment": "Amoxicillin",
            "previous_treatment_response": "Failure",
            "treatments_tried_count": 1
        }
    },
    {
        "name": "6. Golden Retriever - High Severity, Affordable Cost",
        "data": {
            "breed": "Golden Retriever",
            "age": 4,
            "weight": 28.0,
            "medical_history": "None",
            "genetic_predispositions": "None",
            "current_medications": "Beta-blockers",
            "diet": "High-protein",
            "lifestyle": "Active",
            "environment": "Dry",
            "vaccination_status": "Up-to-date",
            "neutering_status": "Not Neutered",
            "living_conditions": "Multi-pet",
            "disease": "Staph Infection",
            "stage": "Severe",
            "severity_score": 85,
            "symptom_count": 8,
            "cost_category": "Affordable"
        }
    },
    {
        "name": "7. Beagle - Fungal with Multiple Symptoms",
        "data": {
            "breed": "Beagle",
            "age": 5,
            "weight": 12.0,
            "medical_history": "None",
            "genetic_predispositions": "Fungal susceptibility",
            "current_medications": "None",
            "diet": "Grain-free",
            "lifestyle": "Outdoor",
            "environment": "Wet",
            "vaccination_status": "Up-to-date",
            "neutering_status": "Neutered",
            "living_conditions": "Multi-pet",
            "disease": "Fungal Infections",
            "stage": "Moderate",
            "severity_score": 65,
            "symptom_count": 7
        }
    },
    {
        "name": "8. Rottweiler - Severe with High Drug Interaction Risk",
        "data": {
            "breed": "Rottweiler",
            "age": 9,
            "weight": 40.0,
            "medical_history": "None",
            "genetic_predispositions": "None",
            "current_medications": "Beta-blockers",
            "diet": "Low-sodium",
            "lifestyle": "Indoor",
            "environment": "Temperate",
            "vaccination_status": "Up-to-date",
            "neutering_status": "Neutered",
            "living_conditions": "Single-pet",
            "disease": "Hypersensitivity Allergic Dermatosis",
            "stage": "Severe",
            "severity_score": 90,
            "symptom_count": 9,
            "drug_interaction_risk": "High"
        }
    },
    {
        "name": "9. Boxer - Bacterial with Expensive Cost Category",
        "data": {
            "breed": "Boxer",
            "age": 6,
            "weight": 28.0,
            "medical_history": "None",
            "genetic_predispositions": "None",
            "current_medications": "None",
            "diet": "Balanced",
            "lifestyle": "Active",
            "environment": "Temperate",
            "vaccination_status": "Up-to-date",
            "neutering_status": "Neutered",
            "living_conditions": "Single-pet",
            "disease": "Bacterial Dermatosis",
            "stage": "Moderate",
            "cost_category": "Expensive"
        }
    },
    {
        "name": "10. Cocker Spaniel - Fungal with Out of Stock Availability",
        "data": {
            "breed": "Cocker Spaniel",
            "age": 4,
            "weight": 12.0,
            "medical_history": "None",
            "genetic_predispositions": "Fungal susceptibility",
            "current_medications": "None",
            "diet": "Hypoallergenic",
            "lifestyle": "Indoor",
            "environment": "Tropical",
            "vaccination_status": "Up-to-date",
            "neutering_status": "Neutered",
            "living_conditions": "Single-pet",
            "disease": "Fungal Infections",
            "stage": "Initial",
            "availability_status": "Out of Stock"
        }
    }
]

if __name__ == "__main__":
    print("="*80)
    print("TESTING API WITH DIFFERENT SCENARIOS")
    print("="*80)
    print("\nMake sure the API server is running: python3 api/app.py")
    print("Press Ctrl+C to stop\n")
    
    # Test health first
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
        else:
            print("⚠️  API server responded but may have issues")
    except:
        print("❌ API server is not running!")
        print("   Start it with: python3 api/app.py")
        exit(1)
    
    # Run all test cases
    for test_case in test_cases:
        test_request(test_case["name"], test_case["data"])
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)

