import requests
import base64
import os
import json

BASE_URL = 'http://localhost:5000'
IMAGE_PATH = 'src/assets/logo.png'

def test_health():
    print("\nTesting /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✓ Health check passed")
            print(f"  Models loaded: {data.get('models_loaded')}")
            print(f"  Configs loaded: {data.get('configs_loaded')}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_predict(model_name):
    print(f"\nTesting /predict with model='{model_name}'...")
    try:
        with open(IMAGE_PATH, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        payload = {
            "image": encoded_string,
            "model": model_name
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✓ Prediction successful for {model_name}")
                print(f"  Model used: {data.get('model_used')}")
                print(f"  Top prediction: {data.get('top_prediction')}")
                return True
            else:
                print(f"✗ Prediction failed: {data.get('error')}")
                return False
        else:
            print(f"✗ Request failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return False

if __name__ == "__main__":
    if test_health():
        test_predict('ResNet50')
        test_predict('EfficientNetB0')
