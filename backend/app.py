"""
Brahmi OCR Backend - Python Flask API
Usage: 
1. Place 'brahmi_model.keras' and 'model_config.json' in the backend/ directory
2. Install dependencies: pip install -r requirements.txt
3. Run: python app.py
4. Backend will run at: http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import json
import io
import base64
import os

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
    'MobileNetV2': os.path.join(BASE_DIR, 'brahmi_model_mobilenetv2.h5')
}
CONFIG_PATHS = {
    'ResNet50': os.path.join(BASE_DIR, 'brahmi_model_resnet50', 'model_config_resnet50.json'),
    'EfficientNetB0': os.path.join(BASE_DIR, 'brahmi_model_efficientnetb0', 'model_config_efficientnetb0.json'),
    'MobileNetV2': os.path.join(BASE_DIR, 'model_config_mobilnetv2.json')
}

# Dictionary of custom objects needed for Keras to deserialize the model
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding
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
        print(f"✓ {model_name} loaded successfully!")
        
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}.")
        print("Please ensure your .keras file is in the backend/ directory.")
    except Exception as e:
        print(f"ERROR loading {model_name}: {e}")

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
        
        print(f"✓ Configuration loaded for {model_name}: {len(class_names)} classes")

    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}.")
    except Exception as e:
        print(f"ERROR loading configuration for {model_name}: {e}")


# ============================================================================
# FLASK API
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

def preprocess_image(img, target_width, target_height):
    """Resizes and normalizes the image for the model."""
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model's expected dimensions
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
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
            # Handle file upload (multipart/form-data)
            file = request.files['image']
            img = Image.open(file.stream)
        
        elif request.json and 'image' in request.json:
            # Handle base64 image (application/json)
            img_data = base64.b64decode(request.json['image'])
            img = Image.open(io.BytesIO(img_data))
        
        else:
            return jsonify({
                'success': False,
                'error': 'No image provided. Send as multipart file or base64 JSON.'
            }), 400

        # --- 2. Determine Model & Predict ---
        model_name = request.form.get('model') or (request.json and request.json.get('model')) or 'ResNet50'
        
        probabilities = None
        class_names = []

        if model_name == 'Ensemble':
            # Ensemble Logic: Soft Voting
            if not models:
                 return jsonify({'success': False, 'error': 'No models loaded for Ensemble.'}), 500
            
            # Use first model's classes as reference (assuming consistency)
            ref_model = list(models.keys())[0]
            class_names = configs[ref_model].get('class_names', [])
            num_classes = len(class_names)
            
            if num_classes == 0:
                 return jsonify({'success': False, 'error': 'Class names configuration missing.'}), 500

            aggregated_probs = np.zeros(num_classes)
            models_used_count = 0
            
            print(f"Running Ensemble on {len(models)} models...")

            for m_name, model in models.items():
                try:
                    cfg = configs.get(m_name, {})
                    # Resize for specific model
                    h = cfg.get('image_height', 64)
                    w = cfg.get('image_width', 256)
                    
                    # Preprocess (resize & normalize)
                    img_arr = preprocess_image(img, w, h)
                    
                    # Predict
                    preds = model.predict(img_arr, verbose=0)[0]
                    probs = tf.nn.softmax(preds).numpy()
                    
                    if len(probs) == num_classes:
                        aggregated_probs += probs
                        models_used_count += 1
                    else:
                        print(f"Skipping {m_name}: Output shape {len(probs)} != {num_classes}")

                except Exception as e:
                    print(f"Error in Ensemble for {m_name}: {e}")

            if models_used_count > 0:
                probabilities = aggregated_probs / models_used_count
            else:
                 return jsonify({'success': False, 'error': 'Ensemble failed. No valid model outputs.'}), 500

        elif model_name in models:
            # Single Model Logic
            model = models[model_name]
            config = configs.get(model_name, {})
            class_names = config.get('class_names', [])
            h = config.get('image_height', 64)
            w = config.get('image_width', 256)
            
            img_arr = preprocess_image(img, w, h)
            preds = model.predict(img_arr, verbose=0)[0]
            probabilities = tf.nn.softmax(preds).numpy()
            
        else:
            return jsonify({
                'success': False,
                'error': f"Model '{model_name}' not available or not loaded."
            }), 400

        # --- 3. Process Results ---
        top_k = 5
        # Ensure we don't go out of bounds if classes < 5
        top_k = min(top_k, len(probabilities))
        
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            char_name = class_names[idx] if idx < len(class_names) else f"Index_{idx}"
            confidence = float(probabilities[idx] * 100)  # Convert to percentage
            
            results.append({
                'character': char_name,
                'confidence': confidence
            })
        
        top_char = results[0]['character']
        top_confidence = results[0]['confidence']
        
        return jsonify({
            'success': True,
            'top_prediction': top_char,
            'top_confidence': top_confidence,
            'predictions': results,
            'model_used': model_name
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f"Internal Server Error: {str(e)}"
        }), 500


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