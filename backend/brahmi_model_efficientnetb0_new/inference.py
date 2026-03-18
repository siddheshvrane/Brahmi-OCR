import torch
from PIL import Image
from torchvision import transforms
from model import EfficientNetB0Classifier
import os

def load_inference_model(weights_path="model_weights.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint to get metadata (classes, idx2label)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    num_classes = checkpoint["num_classes"]
    idx2label = checkpoint["idx2label"]
    
    # Initialize and load model
    model = EfficientNetB0Classifier(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    
    return model, idx2label, device

def predict_character(image_path, model, idx2label, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, pred_idx = torch.max(output, 1)
        
    label = idx2label.get(str(pred_idx.item()), "Unknown")
    return label

if __name__ == "__main__":
    # Example usage
    weights = "model_weights.pth"
    if not os.path.exists(weights):
        print(f"Error: {weights} not found in this folder!")
    else:
        model, idx2label, device = load_inference_model(weights)
        print("Model loaded successfully.")
        
        # Test on an image if provided, else just print readiness
        print("\nTo predict an image, use:")
        print("label = predict_character('path/to/image.jpg', model, idx2label, device)")
