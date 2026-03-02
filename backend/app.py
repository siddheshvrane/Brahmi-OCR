"""
Brahmi OCR Backend - Python Flask API
Usage: 
1. Place 'brahmi_model.keras' and 'model_config.json' in the backend/ directory
2. Install dependencies: pip install -r requirements.txt
3. Run: python app.py
4. Backend will run at: http://127.0.0.1:5000
"""

import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image
import json
import io
import base64
import os
import cv2

# Import segmentation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segmentation import detect_characters, sort_boxes

# ============================================================================
# CUSTOM LAYERS (Required for .keras model loading)
# ============================================================================

@tf.keras.utils.register_keras_serializable(package="BrahmiOCR")
class TransformerBlock(layers.Layer):
    """A standard Transformer encoder block used in your OCR model."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

    def call(self, inputs, training=None):
        # Cast to float32 to avoid type mismatch
        inputs = tf.cast(inputs, tf.float32)
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


@tf.keras.utils.register_keras_serializable(package="BrahmiOCR")
class TokenAndPositionEmbedding(layers.Layer):
    """Layer to generate and add positional embeddings to token embeddings."""
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "embed_dim": self.embed_dim
        })
        return config

    def call(self, inputs):
        # Cast to float32 to avoid type mismatch
        inputs = tf.cast(inputs, tf.float32)
        max_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        position_embeddings = self.position_embedding(positions)
        return inputs + position_embeddings


# ============================================================================
# MODEL LOADING
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    'ResNet50': os.path.join(BASE_DIR, 'brahmi_model_resnet50', 'brahmi_model_resnet50.keras'),
    'EfficientNetB0': os.path.join(BASE_DIR, 'brahmi_model_efficientnetb0', 'brahmi_model_efficientnetb0.keras'),
    'MobileNetV2': os.path.join(BASE_DIR, 'brahmi_model_mobilenet_v1_fixed.keras')
}

CONFIG_PATHS = {
    'ResNet50': os.path.join(BASE_DIR, 'brahmi_model_resnet50', 'model_config_resnet50.json'),
    'EfficientNetB0': os.path.join(BASE_DIR, 'brahmi_model_efficientnetb0', 'model_config_efficientnetb0.json'),
    'MobileNetV2': os.path.join(BASE_DIR, 'model_config_mobilenetv2.json')
}

# Dictionary of custom objects needed for Keras to deserialize the model
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
    'Sequential': tf.keras.Sequential,
    'Functional': tf.keras.Model
}

models = {}
configs = {}

print("Loading Brahmi OCR models...")

for model_name, model_path in MODEL_PATHS.items():
    try:
        print(f"Loading {model_name} from {model_path}...")
        # Load the .keras model with custom objects
        models[model_name] = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False  # Faster loading for inference only
        )
        print(f"OK {model_name} loaded successfully!")
        
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}.")
        print("Please ensure your .keras file is in the backend/ directory.")
    except Exception as e:
        print(f"ERROR loading {model_name}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n[DEBUG] Model loading complete. Models in dict: {list(models.keys())}")
print(f"[DEBUG] MobileNetV2 in models: {'MobileNetV2' in models}\n")

# Load configuration for class names and dimensions
for model_name, config_path in CONFIG_PATHS.items():
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            configs[model_name] = json.load(f)

        class_names = configs[model_name].get('class_names', [])
        img_height = configs[model_name].get('image_height', 64)
        img_width = configs[model_name].get('image_width', 256)

        if not class_names:
            print(f"WARNING: 'class_names' not found in {config_path}.")
        
        print(f"OK Configuration loaded for {model_name}: {len(class_names)} classes")

    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}.")
    except Exception as e:
        print(f"ERROR loading configuration for {model_name}: {e}")


# ============================================================================
# FLASK API
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

def resize_with_padding(img, target_width, target_height, canvas_scale=0.5):
    """
    Resizes image to target dimensions while maintaining aspect ratio using padding.
    canvas_scale: factor to shrink the character within the box to add 'white space' (breathing room).
    """
    # 1. Calculate aspect ratios
    original_w, original_h = img.size
    
    # Scale target dimensions by canvas_scale to create margins
    effective_target_w = int(target_width * canvas_scale)
    effective_target_h = int(target_height * canvas_scale)
    
    ratio_w = effective_target_w / original_w
    ratio_h = effective_target_h / original_h
    
    # Use the smaller ratio to ensure the image fits within scaled target dimensions
    scale = min(ratio_w, ratio_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    
    # 2. Resize maintaining aspect ratio
    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 3. Create a new white background image of the full target size
    new_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    
    # 4. Paste the resized image onto the center of the white background
    offset_x = (target_width - new_w) // 2
    offset_y = (target_height - new_h) // 2
    new_img.paste(resized_img, (offset_x, offset_y))
    
    return new_img


def preprocess_image(img, target_width, target_height):
    """Resizes and normalizes the image for the model."""
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Use letterboxing to preserve aspect ratio
    img = resize_with_padding(img, target_width, target_height)
    
    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    """Predict Brahmi character from uploaded image."""
    try:
        # --- 1. Load and Decode Image ---
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream)
        elif request.json and 'image' in request.json:
            img_data = base64.b64decode(request.json['image'])
            img = Image.open(io.BytesIO(img_data))
        else:
            return jsonify({'success': False, 'error': 'No image provided.'}), 400

        # Ensure RGB (discard alpha channel if present)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # --- 2. Determine Request Params ---
        model_name = request.form.get('model') or (request.json and request.json.get('model')) or 'ResNet50'

        # --- 3. Segmentation ---
        # Convert PIL to OpenCV (RGB -> BGR)
        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        boxes, _ = detect_characters(open_cv_image)
        # boxes = [] # Disabled bounding box logic per user request
        
        if not boxes:
             # Fallback: treat whole image as one char
             boxes = [(0, 0, open_cv_image.shape[1], open_cv_image.shape[0])]
        
        sorted_boxes = sort_boxes(boxes)

        # --- 4. Prediction Logic ---
        results = []
        full_text_array = []
        
        # Helper to get prediction for a batch of crops using specific model
        def get_model_preds(model_key, crop_images):
            model_obj = models[model_key]
            cfg = configs.get(model_key, {})
            # Resize params
            h = cfg.get('image_height', 64)
            w = cfg.get('image_width', 128) # Default 128 for ResNet, others might be 256. 
            
            # Note: app.py previously had 256 default, but ResNet config I saw had 128.
            # We should rely on config or safe default.
            
            batch_arr = []
            for c_img in crop_images:
                # Preprocess: letterbox resize
                c_letterboxed = resize_with_padding(c_img, w, h)
                c_arr = np.array(c_letterboxed, dtype=np.float32)
                
                # Model-specific color space and normalization
                if model_key == 'MobileNetV2':
                    # Reference project used cv2.imread (BGR) and NO normalization
                    c_arr = c_arr[:, :, ::-1]
                else:
                    # Other models (ResNet, EfficientNet) use RGB and [0, 1] normalization
                    c_arr = c_arr / 255.0
                    
                batch_arr.append(c_arr)
            
            batch_input = np.array(batch_arr)
            return model_obj.predict(batch_input, verbose=0)

        # Prepare PIL crops once
        pil_crops = []
        for (x, y, gw, gh) in sorted_boxes:
            pil_crops.append(img.crop((x, y, x+gw, y+gh)))

        final_probabilities = []
        # Get Reference Class Names (from first available model)
        ref_model = list(models.keys())[0] if models else None
        if not ref_model:
             return jsonify({'success': False, 'error': 'No models loaded.'}), 500
        class_names = configs[ref_model].get('class_names', [])

        if model_name == 'Ensemble':
            # Ensemble: Run all models on the batch of crops
            if not models:
                 return jsonify({'success': False, 'error': 'No models for Ensemble.'}), 500
            
            num_crops = len(pil_crops)
            num_classes = len(class_names)
            
            # Shape: (Num_Crops, Num_Classes)
            aggregated_probs = np.zeros((num_crops, num_classes))
            valid_models = 0
            
            print(f"Ensemble: Processing {num_crops} chars with {len(models)} models...")
            
            ensemble_errors = {}
            for m_key in models.keys():
                try:
                    preds = get_model_preds(m_key, pil_crops) # Shape (N, Classes)
                    probs = tf.nn.softmax(preds).numpy()
                    
                    if probs.shape == aggregated_probs.shape:
                        aggregated_probs += probs
                        valid_models += 1
                    else:
                        msg = f"Shape mismatch: {probs.shape} vs {aggregated_probs.shape}"
                        print(msg)
                        ensemble_errors[m_key] = msg
                except Exception as e:
                    msg = str(e)
                    print(f"Error in Ensemble for {m_key}: {msg}")
                    ensemble_errors[m_key] = msg

            if valid_models > 0:
                final_probabilities = aggregated_probs / valid_models
            else:
                 return jsonify({
                     'success': False, 
                     'error': 'Ensemble failed completely.',
                     'details': ensemble_errors
                 }), 500

        elif model_name in models:
            # Single Model
            preds = get_model_preds(model_name, pil_crops)
            final_probabilities = tf.nn.softmax(preds).numpy()
        else:
             return jsonify({'success': False, 'error': f"Model '{model_name}' not found."}), 400

        # --- 5. Decode Results ---
        for i, probs in enumerate(final_probabilities):
            top_idx = np.argmax(probs)
            char_name = class_names[top_idx] if top_idx < len(class_names) else "Unknown"
            conf = float(probs[top_idx] * 100)
            
            full_text_array.append(char_name)
            results.append({
                'character': char_name,
                'confidence': conf,
                'box': sorted_boxes[i]
            })

        top_char = " ".join(full_text_array)
        top_conf = sum([r['confidence'] for r in results]) / len(results) if results else 0.0

        return jsonify({
            'success': True,
            'top_prediction': top_char,
            'top_confidence': top_conf,
            'predictions': results,
            'model_used': model_name
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f"Internal Error: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    loaded_models = list(models.keys())
    return jsonify({
        'status': 'healthy',
        'models_loaded': loaded_models,
        'configs_loaded': list(configs.keys())
    })


@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    return jsonify({
        'service': 'Brahmi OCR API',
        'version': '1.1',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'POST - Predict character from image'
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Brahmi OCR Backend Server Starting...")
    print("="*60)
    print(f"Server running at: http://127.0.0.1:5000")
    print(f"Frontend should connect to: http://localhost:5000")
    print("="*60 + "\n")
    
    # Run server (debug=False for production)
    app.run(host='0.0.0.0', port=5000, debug=True)