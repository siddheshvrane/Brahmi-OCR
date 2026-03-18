import torch
import json
import os

weights_path = r'C:\Siddhesh Project\Brahmi-OCR\backend\brahmi_model_efficientnetb0_new\model_weights.pth'
output_path = r'C:\Siddhesh Project\Brahmi-OCR\backend\brahmi_model_efficientnetb0_new\model_config_efficientnetb0_new.json'

if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    config = {
        'num_classes': checkpoint['num_classes'],
        'idx2label': checkpoint['idx2label'],
        'image_height': 224,
        'image_width': 224
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"Successfully created config at {output_path}")
else:
    print(f"Weights not found at {weights_path}")
