# API Testing Commands

## Quick Start

### 1. Start the API Server

```bash
cd /Users/miksan/Desktop/rp
python3 api/app.py
```

The server will start on **http://localhost:8080**

Keep this terminal open - the server runs in the foreground.

---

## 2. Test Health Endpoint

Open a **new terminal** and run:

```bash
curl http://localhost:8080/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 3. Test Single Prediction

### Using curl:

```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
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

### Using Python test script:

```bash
python3 api/test_api.py
```

---

## 4. Test with Enhanced Features

```bash
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "breed": "Labrador",
    "age": 5,
    "weight": 25.0,
    "medical_history": "None",
    "genetic_predispositions": "None",
    "current_medications": "None",
    "diet": "Grain-free",
    "lifestyle": "Active",
    "environment": "Temperate",
    "vaccination_status": "Up-to-date",
    "neutering_status": "Neutered",
    "living_conditions": "Single-pet",
    "disease": "Fungal Infections",
    "stage": "Initial",
    "severity_score": 45,
    "symptom_count": 3,
    "previous_treatment": "None",
    "previous_treatment_response": "None",
    "treatments_tried_count": 0,
    "drug_allergies": "None",
    "drug_interaction_risk": "Low",
    "cost_category": "Affordable",
    "availability_status": "In Stock"
  }'
```

---

## 5. Test Batch Prediction

```bash
curl -X POST http://localhost:8080/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{
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
  }'
```

---

## Quick Test (All-in-One)

```bash
# Terminal 1: Start server
python3 api/app.py

# Terminal 2: Run tests
python3 api/test_api.py
```

---

## Expected Response Format

```json
{
  "conventional_treatment": "Cephalexin",
  "conventional_confidence": 0.8567,
  "natural_remedies": ["Chamomile", "Aloe vera gel", "Probiotics"],
  "natural_confidences": [0.9234, 0.8765, 0.8123],
  "all_treatment_probabilities": {
    "Amoxicillin": 0.1234,
    "Cephalexin": 0.8567,
    "Fluconazole": 0.0123,
    "Itraconazole": 0.0056,
    "Ketoconazole": 0.0020
  }
}
```

---

## Troubleshooting

**Port already in use:**
```bash
# Find and kill process on port 8080
lsof -ti:8080 | xargs kill -9
```

**Model not loaded:**
- Check that `models/saved/best_model.pth` exists
- Check that `models/saved/preprocessor.pkl` exists

**Connection refused:**
- Make sure the API server is running
- Check the port (should be 8080)

