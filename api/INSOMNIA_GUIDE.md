# Insomnia API Request Format Guide

## Setup

**Base URL:** `http://localhost:8080`

---

## 1. Health Check Request

### Request Configuration:
- **Method:** `GET`
- **URL:** `http://localhost:8080/health`
- **Headers:** None required
- **Body:** None

### Insomnia Setup:
1. Create new request
2. Set method to **GET**
3. Set URL to: `http://localhost:8080/health`
4. Click **Send**

### Expected Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

---

## 2. Single Prediction Request

### Request Configuration:
- **Method:** `POST`
- **URL:** `http://localhost:8080/predict`
- **Headers:**
  - `Content-Type: application/json`
- **Body:** JSON (see below)

### Insomnia Setup:

1. **Create new request**
   - Name: "Predict Treatment"

2. **Set Method:** `POST`

3. **Set URL:** `http://localhost:8080/predict`

4. **Add Header:**
   - Key: `Content-Type`
   - Value: `application/json`

5. **Set Body:**
   - Select **Body** tab
   - Choose **JSON** format
   - Paste the following JSON:

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

6. **Click Send**

### Expected Response:
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

---

## 3. Batch Prediction Request

### Request Configuration:
- **Method:** `POST`
- **URL:** `http://localhost:8080/predict/batch`
- **Headers:**
  - `Content-Type: application/json`
- **Body:** JSON (see below)

### Insomnia Setup:

1. **Create new request**
   - Name: "Batch Predict"

2. **Set Method:** `POST`

3. **Set URL:** `http://localhost:8080/predict/batch`

4. **Add Header:**
   - Key: `Content-Type`
   - Value: `application/json`

5. **Set Body:**
   - Select **Body** tab
   - Choose **JSON** format
   - Paste the following JSON:

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

6. **Click Send**

### Expected Response:
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

## Step-by-Step Insomnia Setup

### Creating a New Request:

1. **Open Insomnia**
2. **Create a new folder** (optional): Right-click → "New Folder" → Name it "Dog Treatment API"
3. **Create a new request:**
   - Right-click folder (or workspace) → "New Request"
   - Name it (e.g., "Predict Treatment")

### Configuring the Request:

1. **Method Dropdown:** Select `GET` or `POST`
2. **URL Bar:** Enter `http://localhost:8080/[endpoint]`
3. **Headers Tab:**
   - Click "Add Header"
   - For POST requests, add:
     - Key: `Content-Type`
     - Value: `application/json`
4. **Body Tab:**
   - Select "JSON" from dropdown
   - Paste your JSON payload
5. **Click "Send"** button

---

## Example Request Templates

### Template 1: Chihuahua with Severe Allergic Dermatosis
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

### Template 2: Labrador with Fungal Infection
```json
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
```

### Template 3: German Shepherd with Staph Infection
```json
{
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
  "stage": "Moderate"
}
```

---

## Valid Input Values Reference

| Field | Valid Values |
|-------|-------------|
| `breed` | Any dog breed (e.g., "Chihuahua", "Labrador", "Poodle", "German Shepherd") |
| `age` | Any positive number (years) |
| `weight` | Any positive number (kilograms) |
| `medical_history` | "None", "Eye infections", "Skin allergies", "Allergy to Penicillin" |
| `genetic_predispositions` | "None", "Dermatitis", "Fungal susceptibility" |
| `current_medications` | "None", "Heartworm Preventative", "Beta-blockers", "Insulin", "Antihistamines" |
| `diet` | "Balanced", "Grain-free", "Hypoallergenic", "High-protein", "Low-sodium" |
| `lifestyle` | "Indoor", "Outdoor", "Active" |
| `environment` | "Temperate", "Tropical", "Cold", "Humid", "Wet", "Dry" |
| `vaccination_status` | "Up-to-date", "Not Up-to-date" |
| `neutering_status` | "Neutered", "Not Neutered" |
| `living_conditions` | "Single-pet", "Multi-pet" |
| `disease` | "Bacterial Dermatosis", "Fungal Infections", "Hypersensitivity Allergic Dermatosis", "Staph Infection" |
| `stage` | "Initial", "Mild", "Moderate", "Severe" |

---

## Troubleshooting

### Error: "Connection refused"
- Make sure the API server is running: `python3 api/app.py`
- Check the URL is correct: `http://localhost:8080`

### Error: "Missing required fields"
- Make sure all 14 fields are included in the JSON body
- Check for typos in field names (they must match exactly)

### Error: "Model not loaded"
- Make sure `models/saved/best_model.pth` exists
- Make sure `models/saved/preprocessor.pkl` exists

### Response is empty or error
- Check the server terminal for error messages
- Verify JSON is valid (use a JSON validator)
- Make sure `Content-Type: application/json` header is set

---

## Quick Copy-Paste for Insomnia

### Health Check:
```
GET http://localhost:8080/health
```

### Single Prediction:
```
POST http://localhost:8080/predict
Content-Type: application/json

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

