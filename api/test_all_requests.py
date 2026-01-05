"""
Comprehensive API Testing Script
Tests the API with 20 different scenarios to get diverse responses
"""
import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:8080"

def test_request(name, data, expected_treatment_type=None):
    """Test a single request and display results."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print('='*80)
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Status: Success")
            print(f"\n📋 Input Summary:")
            print(f"   Breed: {data.get('breed')}")
            print(f"   Disease: {data.get('disease')}")
            print(f"   Stage: {data.get('stage')}")
            if 'drug_allergies' in data:
                print(f"   Drug Allergies: {data.get('drug_allergies')}")
            if 'previous_treatment' in data:
                print(f"   Previous Treatment: {data.get('previous_treatment')} ({data.get('previous_treatment_response')})")
            
            print(f"\n💊 Conventional Treatment:")
            print(f"   Treatment: {result.get('conventional_treatment')}")
            print(f"   Confidence: {result.get('conventional_confidence', 0):.2%}")
            
            print(f"\n🌿 Natural Remedies:")
            remedies = result.get('natural_remedies', [])
            confidences = result.get('natural_confidences', [])
            if remedies:
                for remedy, conf in zip(remedies, confidences):
                    print(f"   - {remedy}: {conf:.2%}")
            else:
                print("   None recommended")
            
            print(f"\n📊 All Treatment Probabilities:")
            probs = result.get('all_treatment_probabilities', {})
            for treatment, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                marker = " ⭐" if treatment == result.get('conventional_treatment') else ""
                print(f"   {treatment}: {prob:.2%}{marker}")
            
            if 'top_2_predictions' in result:
                print(f"\n🔝 Top 2 Predictions:")
                for treatment, prob in result['top_2_predictions'].items():
                    print(f"   {treatment}: {prob:.2%}")
            
            return result
        else:
            print(f"❌ Status: Error {response.status_code}")
            print(f"   Message: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to API server.")
        print("   Make sure the server is running: python3 api/app.py")
        return None
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None

def main():
    """Run all test requests."""
    print("="*80)
    print("COMPREHENSIVE API TESTING - 20 DIFFERENT SCENARIOS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        return
    
    # Load test requests
    try:
        with open('api/COMPREHENSIVE_TEST_REQUESTS.json', 'r') as f:
            test_data = json.load(f)
        test_cases = test_data['test_requests']
    except FileNotFoundError:
        print("❌ Test requests file not found!")
        return
    
    # Track results
    results = []
    successful = 0
    failed = 0
    
    # Run all test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n[{i}/{len(test_cases)}]")
        result = test_request(test_case['name'], test_case['request'])
        
        if result:
            results.append({
                'name': test_case['name'],
                'treatment': result.get('conventional_treatment'),
                'confidence': result.get('conventional_confidence', 0)
            })
            successful += 1
        else:
            failed += 1
        
        time.sleep(0.3)  # Small delay between requests
    
    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    # Show treatment diversity
    print(f"\n📊 Treatment Diversity:")
    treatments = {}
    for r in results:
        treatment = r['treatment']
        treatments[treatment] = treatments.get(treatment, 0) + 1
    
    for treatment, count in sorted(treatments.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100 if results else 0
        print(f"   {treatment}: {count} times ({percentage:.1f}%)")
    
    print(f"\n✅ Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()

