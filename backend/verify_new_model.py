import torch
import sys
import os
import numpy as np
from PIL import Image

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brahmi_model_efficientnetb0_new.model import EfficientNetB0Classifier

def test_loading():
    print("Testing model loading...")
    weights_path = r'C:\Siddhesh Project\Brahmi-OCR\backend\brahmi_model_efficientnetb0_new\model_weights.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        num_classes = checkpoint["num_classes"]
        idx2label = checkpoint["idx2label"]
        
        model = EfficientNetB0Classifier(num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully with {num_classes} classes.")
        
        # Test dummy prediction
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Prediction output shape: {output.shape}")
            
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_loading():
        print("\nVerification PASSED!")
    else:
        print("\nVerification FAILED!")
