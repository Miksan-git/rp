#!/bin/bash
# Quick test script - Copy and paste these commands

API_URL="http://localhost:8080"

echo "Testing API with different requests..."
echo ""

# Test 1: Your original request
echo "Test 1: Bulldog - Hypersensitivity"
curl -X POST $API_URL/predict \
  -H 'Content-Type: application/json' \
  -d '{
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
  }' | python3 -m json.tool

echo -e "\n\n"

# Test 2: Fungal infection
echo "Test 2: Labrador - Fungal Infection"
curl -X POST $API_URL/predict \
  -H 'Content-Type: application/json' \
  -d '{
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
  }' | python3 -m json.tool

echo -e "\n\n"

# Test 3: With drug allergy
echo "Test 3: Chihuahua - With Penicillin Allergy"
curl -X POST $API_URL/predict \
  -H 'Content-Type: application/json' \
  -d '{
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
    "drug_allergies": "Penicillin"
  }' | python3 -m json.tool

