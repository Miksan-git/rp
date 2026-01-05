# API Request and Response Examples

## 1. Single Prediction Endpoint

### Request

**Endpoint:** `POST http://localhost:5000/predict`

**Headers:**
```
Content-Type: application/json
```

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

### Response (Success - 200 OK)

```json
{
  "conventional_treatment": "Amoxicillin",
  "conventional_confidence": 0.4523,
  "natural_remedies": [
    "Chamomile",
    "Aloe vera gel",
    "Probiotics",
    "Omega-3"
  ],
  "natural_confidences": [
    0.9234,
    0.8845,
    0.7654,
    0.7123
  ],
  "all_treatment_probabilities": {
    "Amoxicillin": 0.4523,
    "Cephalexin": 0.3845,
    "Fluconazole": 0.0821,
    "Itraconazole": 0.0512,
    "Ketoconazole": 0.0299
  }
}
```

### Response Fields Explanation:

- **conventional_treatment**: The predicted conventional veterinary treatment (string)
- **conventional_confidence**: Confidence score for the conventional treatment (0.0 to 1.0)
- **natural_remedies**: List of recommended natural/home remedies (array of strings)
- **natural_confidences**: Confidence scores for each natural remedy (array of floats, 0.0 to 1.0)
- **all_treatment_probabilities**: Probability distribution across all possible conventional treatments (object)

---

## 2. Batch Prediction Endpoint

### Request

**Endpoint:** `POST http://localhost:5000/predict/batch`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
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
```

### Response (Success - 200 OK)

```json
{
  "results": [
    {
      "conventional_treatment": "Amoxicillin",
      "conventional_confidence": 0.4523,
      "natural_remedies": [
        "Chamomile",
        "Aloe vera gel",
        "Probiotics"
      ]
    },
    {
      "conventional_treatment": "Ketoconazole",
      "conventional_confidence": 0.8234,
      "natural_remedies": [
        "Tea tree oil",
        "Coconut oil",
        "Calendula"
      ]
    }
  ]
}
```

---

## 3. Health Check Endpoint

### Request

**Endpoint:** `GET http://localhost:5000/health`

**No body required**

### Response (Success - 200 OK)

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

---

## 4. Error Responses

### Missing Required Fields (400 Bad Request)

**Request:**
```json
{
  "breed": "Chihuahua",
  "age": 6
  // Missing other required fields
}
```

**Response:**
```json
{
  "error": "Missing required fields: weight, medical_history, genetic_predispositions, current_medications, diet, lifestyle, environment, vaccination_status, neutering_status, living_conditions, disease, stage"
}
```

### Model Not Loaded (500 Internal Server Error)

**Response:**
```json
{
  "error": "Model not loaded. Please ensure model files exist."
}
```

### Invalid JSON (400 Bad Request)

**Response:**
```json
{
  "error": "No JSON data provided"
}
```

### Prediction Failed (500 Internal Server Error)

**Response:**
```json
{
  "error": "Prediction failed: [error message]"
}
```

---

## 5. Valid Input Values

### Required Fields and Valid Values:

| Field | Type | Valid Values |
|-------|------|--------------|
| `breed` | string | Any dog breed (e.g., "Chihuahua", "Labrador", "Poodle") |
| `age` | number | Any positive number (years) |
| `weight` | number | Any positive number (kilograms) |
| `medical_history` | string | "None", "Eye infections", "Skin allergies", "Allergy to Penicillin" |
| `genetic_predispositions` | string | "None", "Dermatitis", "Fungal susceptibility" |
| `current_medications` | string | "None", "Heartworm Preventative", "Beta-blockers", "Insulin", "Antihistamines" |
| `diet` | string | "Balanced", "Grain-free", "Hypoallergenic", "High-protein", "Low-sodium" |
| `lifestyle` | string | "Indoor", "Outdoor", "Active" |
| `environment` | string | "Temperate", "Tropical", "Cold", "Humid", "Wet", "Dry" |
| `vaccination_status` | string | "Up-to-date", "Not Up-to-date" |
| `neutering_status` | string | "Neutered", "Not Neutered" |
| `living_conditions` | string | "Single-pet", "Multi-pet" |
| `disease` | string | "Bacterial Dermatosis", "Fungal Infections", "Hypersensitivity Allergic Dermatosis", "Staph Infection" |
| `stage` | string | "Initial", "Mild", "Moderate", "Severe" |

---

## 6. Example Using cURL

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

## 7. Example Using Python

```python
import requests

# Single prediction
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

result = response.json()
print(f"Conventional Treatment: {result['conventional_treatment']}")
print(f"Confidence: {result['conventional_confidence']}")
print(f"Natural Remedies: {result['natural_remedies']}")
```

## 8. Example Using JavaScript (Fetch API)

```javascript
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    breed: "Chihuahua",
    age: 6,
    weight: 2.8,
    medical_history: "None",
    genetic_predispositions: "None",
    current_medications: "None",
    diet: "Balanced",
    lifestyle: "Indoor",
    environment: "Temperate",
    vaccination_status: "Not Up-to-date",
    neutering_status: "Neutered",
    living_conditions: "Multi-pet",
    disease: "Hypersensitivity Allergic Dermatosis",
    stage: "Severe"
  })
})
.then(response => response.json())
.then(data => {
  console.log('Conventional Treatment:', data.conventional_treatment);
  console.log('Confidence:', data.conventional_confidence);
  console.log('Natural Remedies:', data.natural_remedies);
})
.catch(error => console.error('Error:', error));
```

