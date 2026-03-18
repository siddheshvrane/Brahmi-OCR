import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import json
import sys
from pathlib import Path

# Import the model architecture
from model import ResNet50Classifier

class BrahmiClassifier:
    def __init__(self, model_path, class_names_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load class names
        with open(class_names_path, 'r', encoding='utf-8') as f:
            self.class_names = json.load(f)
        
        self.num_classes = len(self.class_names)
        
        # Initialize model
        self.model = ResNet50Classifier(num_classes=self.num_classes)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, image_path, top_k=5):
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Inference
        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        
        # Get top-k
        top_probs, top_indices = torch.topk(probs, k=top_k)
        
        results = []
        for i in range(top_k):
            idx = top_indices[i].item()
            results.append({
                "character": self.class_names[idx],
                "confidence": top_probs[i].item()
            })
        
        return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    # Initialize classifier
    classifier = BrahmiClassifier(
        model_path="resnet50_brahmi.pth",
        class_names_path="class_names.json"
    )
    
    # Predict
    print(f"\nPredicting: {image_path}")
    print("-" * 30)
    try:
        predictions = classifier.predict(image_path)
        for i, res in enumerate(predictions, 1):
            print(f"{i}. {res['character']:10s} : {res['confidence']*100:>6.2f}%")
        
        print("-" * 30)
        print(f"Final Prediction: {predictions[0]['character']}")
    except Exception as e:
        print(f"Error: {e}")
