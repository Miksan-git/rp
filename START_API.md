# How to Run the API

## Step 1: Install Dependencies

If Flask is not installed, run:
```bash
pip install flask flask-cors
```

Or install all API requirements:
```bash
pip install -r api/requirements.txt
```

## Step 2: Start the API Server

From the project root directory (`/Users/miksan/Desktop/rp`), run:

```bash
python api/app.py
```

You should see output like:
```
Loading model...
Using device: cpu
Preprocessor loaded successfully
Model loaded successfully
Model has 586255 parameters

================================================================================
API Server Starting...
================================================================================
Endpoints:
  GET  /health - Health check
  POST /predict - Single prediction
  POST /predict/batch - Batch predictions

Server running on http://localhost:5000
================================================================================

 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://localhost:5000
```

**Keep this terminal window open** - the server is now running!

## Step 3: Test the API

### Option 1: Use the Test Script (Easiest)

Open a **new terminal window** and run:

```bash
cd /Users/miksan/Desktop/rp
python api/test_api.py
```

This will test all endpoints automatically.

### Option 2: Use cURL

In a new terminal:

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
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

### Option 3: Use Python

Create a test file `test_api_manual.py`:

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

Run it:
```bash
python test_api_manual.py
```

## Step 4: Stop the Server

To stop the API server, go back to the terminal where it's running and press:
```
Ctrl + C
```

## Troubleshooting

### Error: "Model not loaded"
- Make sure `models/saved/best_model.pth` exists
- Make sure `models/saved/preprocessor.pkl` exists
- Check that training completed successfully

### Error: "ModuleNotFoundError: No module named 'flask'"
- Install Flask: `pip install flask flask-cors`

### Error: "Address already in use"
- Port 5000 is already in use
- Either stop the other process or change the port in `api/app.py` (last line)

### Error: "Connection refused"
- Make sure the API server is running
- Check that you're using the correct URL: `http://localhost:5000`

## Quick Start Summary

```bash
# Terminal 1: Start the server
cd /Users/miksan/Desktop/rp
python api/app.py

# Terminal 2: Test the API
cd /Users/miksan/Desktop/rp
python api/test_api.py
```

That's it! The API is now running and ready to use.

