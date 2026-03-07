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
import onnxruntime as ort

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
    'MobileNetV2': os.path.join(BASE_DIR, 'brahmi_model_mobilenet_v2', 'brahmi_ocr_best.onnx')
}

CONFIG_PATHS = {
    'ResNet50': os.path.join(BASE_DIR, 'brahmi_model_resnet50', 'model_config_resnet50.json'),
    'EfficientNetB0': os.path.join(BASE_DIR, 'brahmi_model_efficientnetb0', 'model_config_efficientnetb0.json'),
    'MobileNetV2': os.path.join(BASE_DIR, 'brahmi_model_mobilenet_v2', 'brahmi_ocr_best_config.json')
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
translit_mapping = {}

# Load transliteration mapping
try:
    mapping_path = os.path.join(BASE_DIR, 'transliteration_mapping.json')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            translit_mapping = json.load(f)
        print(f"OK Transliteration mapping loaded: {len(translit_mapping)} entries")
    else:
        print("WARNING: transliteration_mapping.json not found. Run generate_mapping.py first.")
except Exception as e:
    print(f"ERROR loading transliteration mapping: {e}")

print("Loading Brahmi OCR models...")

for model_name, model_path in MODEL_PATHS.items():
    try:
        print(f"Loading {model_name} from {model_path}...")
        if model_path.endswith('.onnx'):
            # Load ONNX model
            models[model_name] = ort.InferenceSession(model_path)
            print(f"OK {model_name} ONNX session loaded successfully!")
        else:
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

        # Load configuration for class names
        if 'idx2label' in configs[model_name]:
            idx2label = configs[model_name]['idx2label']
            max_id = max(int(k) for k in idx2label.keys())
            class_names = [idx2label.get(str(i), '<UNK>') for i in range(max_id + 1)]
            configs[model_name]['class_names'] = class_names
        elif 'idx2char' in configs[model_name]:
            idx2char = configs[model_name]['idx2char']
            max_id = max(int(k) for k in idx2char.keys())
            class_names = [idx2char.get(str(i), '<UNK>') for i in range(max_id + 1)]
            configs[model_name]['class_names'] = class_names
        elif 'id2char' in configs[model_name]:
            id2char = configs[model_name]['id2char']
            max_id = max(int(k) for k in id2char.keys())
            class_names = [id2char.get(str(i), '<UNK>') for i in range(max_id + 1)]
            configs[model_name]['class_names'] = class_names

        class_names = configs[model_name].get('class_names', [])
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

def resize_with_padding(img, target_width, target_height, padding_percent=0.15):
    """
    Resizes image to target dimensions utilizing 'Padding to Square' method:
    1. Calculate the largest dimension of the crop.
    2. Add extra padding on all sides (e.g., 15%) to mimic training data constraints.
    3. Pad the original image with a white background to a perfect square.
    4. Resize perfectly to target_width x target_height.
    """
    original_w, original_h = img.size
    max_dim = max(original_w, original_h)
    
    # Calculate padding based on the largest dimension
    padding = int(max_dim * padding_percent)
    
    # New square dimension
    new_dim = max_dim + 2 * padding
    
    # Create a new white background image of the perfect square size
    square_img = Image.new("RGB", (new_dim, new_dim), (255, 255, 255))
    
    # Paste the original crop onto the center of this white square
    offset_x = (new_dim - original_w) // 2
    offset_y = (new_dim - original_h) // 2
    square_img.paste(img, (offset_x, offset_y))
    
    # Resize the perfectly square image to the model's target dimensions
    # This mimics PyTorch transforms.Resize((128, 128)) behavior.
    final_img = square_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    return final_img


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

def roman_to_devanagari(label):
    """Returns Devanagari transliteration from pre-loaded mapping."""
    return translit_mapping.get(label, label)


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
        transliteration = request.form.get('transliteration') or (request.json and request.json.get('transliteration')) or 'latin'

        # --- 3. Segmentation ---
        # Convert PIL to OpenCV (RGB -> BGR)
        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        boxes, _ = detect_characters(open_cv_image)
        
        # If 0 or 1 character is detected, use the full image as the single character box.
        # This provides the character with the most context and avoids tight-edge crops.
        if len(boxes) <= 1:
             sorted_boxes = [(0, 0, open_cv_image.shape[1], open_cv_image.shape[0])]
        else:
             # For multiple characters, use the detected bounding boxes with padding logic
             sorted_boxes = sort_boxes(boxes)

        # --- 4. Prediction Logic ---
        results = []
        full_text_array = []
        
        # Helper to get prediction for a batch of crops using specific model
        def get_model_preds(model_key, crop_images):
            model_obj = models[model_key]
            cfg = configs.get(model_key, {})
            is_onnx = MODEL_PATHS.get(model_key, "").endswith('.onnx')
            
            # Resize params — read from config so it works for any model size
            h = cfg.get('image_height', 224)
            w = cfg.get('image_width', 224)
            
            # ImageNet normalization constants (mean/std used in PyTorch training)
            IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            batch_arr = []
            for c_img in crop_images:
                # Preprocess: letterbox resize
                c_letterboxed = resize_with_padding(c_img, w, h)
                c_arr = np.array(c_letterboxed, dtype=np.float32)
                
                # Model-specific color space and normalization
                if is_onnx:
                    # ONNX MobileNetV2 classifier (PyTorch): CHW layout + ImageNet normalization
                    c_arr = c_arr / 255.0
                    c_arr = (c_arr - IMAGENET_MEAN) / IMAGENET_STD   # normalize channels
                    c_arr = np.transpose(c_arr, (2, 0, 1))           # HWC -> CHW
                else:
                    # Keras models (ResNet, EfficientNet): RGB [0,1] normalization
                    c_arr = c_arr / 255.0
                    
                batch_arr.append(c_arr)
            
            batch_input = np.array(batch_arr)
            
            if is_onnx:
                ort_inputs = {'image': batch_input}
                logits = model_obj.run(None, ort_inputs)[0]
                # Apply Temperature Scaling (T=0.2) to sharpen confidence.
                # PyTorch training with label_smoothing=0.1 and weight_decay
                # causes the raw logits range to be small, resulting in <10% 
                # confidence after Softmax, even when predicting perfectly.
                logits = logits / 0.2
                return logits
                
            return model_obj.predict(batch_input, verbose=0)

        # Prepare PIL crops once
        pil_crops = []
        for (x, y, gw, gh) in sorted_boxes:
            pil_crops.append(img.crop((x, y, x+gw, y+gh)))

        # Get Reference Class Names for decoding
        # If Ensemble, we use the labels from any available model since they are now unified
        # If Single Model, we use that model's specific labels
        label_model = model_name if model_name in configs else list(configs.keys())[0]
        if not label_model:
             return jsonify({'success': False, 'error': 'No model configurations loaded.'}), 500
        class_names = configs[label_model].get('class_names', [])

        if model_name == 'Ensemble':
            # Ensemble: Run all models and pick the best for each character
            if not models:
                 return jsonify({'success': False, 'error': 'No models for Ensemble.'}), 500
            
            num_crops = len(pil_crops)
            num_classes = len(class_names)
            
            # Store probabilities from each model
            all_model_probs = []
            
            print(f"Ensemble (Per-Char Max-Conf): Processing {num_crops} chars with {len(models)} models...")
            
            ensemble_errors = {}
            for m_key in models.keys():
                try:
                    preds = get_model_preds(m_key, pil_crops)
                    probs = tf.nn.softmax(preds).numpy()
                    
                    if probs.shape[1] == num_classes:
                        all_model_probs.append(probs)
                    else:
                        msg = f"Shape mismatch for {m_key}: {probs.shape[1]} classes vs {num_classes}"
                        print(msg)
                        ensemble_errors[m_key] = msg
                except Exception as e:
                    msg = str(e)
                    print(f"Error in Ensemble for {m_key}: {msg}")
                    ensemble_errors[m_key] = msg

            if all_model_probs:
                # Per-character maximum confidence selection
                final_probabilities = np.zeros((num_crops, num_classes))
                for i in range(num_crops):
                    best_m_idx = 0
                    max_val = -1.0
                    for m_idx, m_probs in enumerate(all_model_probs):
                        # Find the model that is most confident about its top prediction for this character
                        current_max = np.max(m_probs[i])
                        if current_max > max_val:
                            max_val = current_max
                            best_m_idx = m_idx
                    final_probabilities[i] = all_model_probs[best_m_idx][i]
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
        full_text_latin = []
        full_text_devanagari = []

        for i, probs in enumerate(final_probabilities):
            top_idx = np.argmax(probs)
            char_name_latin = class_names[top_idx] if top_idx < len(class_names) else "Unknown"
            char_name_devanagari = roman_to_devanagari(char_name_latin)
            conf = float(probs[top_idx] * 100)
            
            full_text_latin.append(char_name_latin)
            full_text_devanagari.append(char_name_devanagari)

            results.append({
                'character': char_name_latin,
                'character_devanagari': char_name_devanagari,
                'confidence': conf,
                'box': sorted_boxes[i]
            })

        top_char_latin = " ".join(full_text_latin)
        top_char_devanagari = " ".join(full_text_devanagari)
        top_conf = sum([r['confidence'] for r in results]) / len(results) if results else 0.0

        return jsonify({
            'success': True,
            'top_prediction': top_char_latin,
            'top_prediction_devanagari': top_char_devanagari,
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