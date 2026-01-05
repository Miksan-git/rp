# Dog Treatment Recommendation API

REST API for testing the trained dog treatment recommendation model.

## Installation

1. Install API dependencies:
```bash
cd api
pip install -r requirements.txt
```

Or from project root:
```bash
pip install flask flask-cors
```

## Running the API

From the project root directory:
```bash
python api/app.py
```

The API will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API and model are loaded correctly.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### 2. Single Prediction
**POST** `/predict`

Get treatment recommendations for a single dog profile.

**Request Body:**
```json
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
```

**Response:**
```json
{
  "conventional_treatment": "Amoxicillin",
  "conventional_confidence": 0.4523,
  "natural_remedies": ["Chamomile", "Aloe vera gel", "Probiotics"],
  "natural_confidences": [0.92, 0.88, 0.76],
  "all_treatment_probabilities": {
    "Amoxicillin": 0.4523,
    "Cephalexin": 0.3845,
    "Fluconazole": 0.0821,
    "Itraconazole": 0.0512,
    "Ketoconazole": 0.0299
  }
}
```

### 3. Batch Prediction
**POST** `/predict/batch`

Get treatment recommendations for multiple dog profiles.

**Request Body:**
```json
{
  "profiles": [
    {
      "breed": "Chihuahua",
      "age": 6,
      ...
    },
    {
      "breed": "Labrador",
      "age": 5,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "conventional_treatment": "Amoxicillin",
      "conventional_confidence": 0.4523,
      "natural_remedies": ["Chamomile", "Aloe vera gel"]
    },
    {
      "conventional_treatment": "Ketoconazole",
      "conventional_confidence": 0.8234,
      "natural_remedies": ["Tea tree oil", "Coconut oil"]
    }
  ]
}
```

## Testing the API

### Using the test script:
```bash
python api/test_api.py
```

### Using curl:

**Health check:**
```bash
curl http://localhost:5000/health
```

**Single prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Using Python requests:
```python
import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={
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
)

print(response.json())
```

## Valid Input Values

### Breed:
Any dog breed (e.g., "Chihuahua", "Labrador", "Poodle", "German Shepherd", etc.)

### Age:
Numeric value in years (e.g., 1, 5, 10)

### Weight:
Numeric value in kilograms (e.g., 2.8, 25.0, 35.5)

### Medical History:
- "None"
- "Eye infections"
- "Skin allergies"
- "Allergy to Penicillin"

### Genetic Predispositions:
- "None"
- "Dermatitis"
- "Fungal susceptibility"

### Current Medications:
- "None"
- "Heartworm Preventative"
- "Beta-blockers"
- "Insulin"
- "Antihistamines"

### Diet:
- "Balanced"
- "Grain-free"
- "Hypoallergenic"
- "High-protein"
- "Low-sodium"

### Lifestyle:
- "Indoor"
- "Outdoor"
- "Active"

### Environment:
- "Temperate"
- "Tropical"
- "Cold"
- "Humid"
- "Wet"
- "Dry"

### Vaccination Status:
- "Up-to-date"
- "Not Up-to-date"

### Neutering Status:
- "Neutered"
- "Not Neutered"

### Living Conditions:
- "Single-pet"
- "Multi-pet"

### Disease:
- "Bacterial Dermatosis"
- "Fungal Infections"
- "Hypersensitivity Allergic Dermatosis"
- "Staph Infection"

### Stage:
- "Initial"
- "Mild"
- "Moderate"
- "Severe"

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing or invalid fields)
- `500`: Server error (model not loaded or prediction failed)

Error response format:
```json
{
  "error": "Error message description"
}
```

