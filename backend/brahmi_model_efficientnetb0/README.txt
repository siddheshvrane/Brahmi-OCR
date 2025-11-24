======================================================================
BRAHMI OCR MODEL PACKAGE
======================================================================

📦 PACKAGE CONTENTS:
----------------------------------------------------------------------
• temp_current_model.keras - Trained Keras model
• class_names.json - Character class mappings
• model_config.json - Model configuration
• model_summary.txt - Model architecture details
• inference_example.py - Usage example code
• README.txt - This file

📊 MODEL SPECIFICATIONS:
----------------------------------------------------------------------
Architecture: ResNet50 + Transformer
Input Size: 64x128x3
Number of Classes: 214
Total Parameters: 5,299,961
Transformer Layers: 2
Embedding Dimension: 128

🚀 QUICK START:
----------------------------------------------------------------------
1. Install dependencies:
   pip install tensorflow pillow numpy

2. Load the model:
   import tensorflow as tf
   import json

   model = tf.keras.models.load_model('temp_current_model.keras')
   with open('class_names.json', 'r') as f:
       class_names = json.load(f)

3. See inference_example.py for complete usage

⚠️ IMPORTANT NOTES:
----------------------------------------------------------------------
• Images must be preprocessed to same size as training
• Normalize pixel values to [0, 1] range
• Model uses logits output (apply softmax for probabilities)
• Custom layers require special loading (see example)

📅 Created: 2025-11-19 15:09:03
======================================================================
