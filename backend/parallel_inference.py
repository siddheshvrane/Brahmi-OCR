#!/usr/bin/env python3
"""
Brahmi OCR Parallel Inference
Usage: python parallel_inference.py image.png
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import sys
import cv2
import os
from PIL import Image

# Import segmentation module
# Ensure backend is in path if running from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segmentation import detect_characters, sort_boxes

# ============================================================================
# CUSTOM LAYERS (Required for loading model)
# ============================================================================

@tf.keras.utils.register_keras_serializable(package="BrahmiOCR")
class TransformerBlock(layers.Layer):
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
        config.update({"embed_dim": self.embed_dim,
                       "num_heads": self.num_heads,
                       "ff_dim": self.ff_dim,
                       "rate": self.rate,})
        return config

    def call(self, inputs, training=None):
        inputs = tf.cast(inputs, tf.float32)
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


@tf.keras.utils.register_keras_serializable(package="BrahmiOCR")
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len,
                       "embed_dim": self.embed_dim,})
        return config

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        max_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        position_embeddings = self.position_embedding(positions)
        return inputs + position_embeddings


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_recognition_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, 'brahmi_model_resnet50', 'model_config_resnet50.json')
    class_names_path = os.path.join(base_path, 'brahmi_model_resnet50', 'class_names_resnet50.json')
    model_path = os.path.join(base_path, 'brahmi_model_resnet50', 'brahmi_model_resnet50.keras')

    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Load class names
    with open(class_names_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    # Load model with custom objects
    custom_objects = {
        'TransformerBlock': TransformerBlock,
        'TokenAndPositionEmbedding': TokenAndPositionEmbedding
    }

    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False
    )
    
    return model, config, class_names

# Initialize model globally to avoid reloading
MODEL, CONFIG, CLASS_NAMES = load_recognition_model()

def recognize_inscription(image_path):
    """
    Recognizes all characters in an inscription.
    """
    # 1. Detect characters
    boxes, original_img = detect_characters(image_path)
    if not boxes:
        return "No characters detected", []

    # 2. Sort characters
    sorted_boxes = sort_boxes(boxes)

    # 3. Prepare Batch
    batch_images = []
    
    target_w = CONFIG['image_width']
    target_h = CONFIG['image_height']

    for (x, y, w, h) in sorted_boxes:
        # Crop
        roi = original_img[y:y+h, x:x+w]
        
        # Resize/Preprocess
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        
        # Resize with padding or stretching? Using simple resize for now to match training commonly
        # Ideally, we should pad to aspect ratio, but let's stick to what inference_example did
        pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        img_array = np.array(pil_img, dtype=np.float32) / 255.0
        img_array = np.clip(img_array, 0.0, 1.0)
        
        batch_images.append(img_array)

    batch_input = np.array(batch_images)

    # 4. Batch Inference
    # verbose=0 to silence progress bar
    predictions = MODEL.predict(batch_input, verbose=0)

    # 5. Decode results
    full_text = []
    details = []

    for i, pred in enumerate(predictions):
        top_idx = np.argmax(pred)
        char = CLASS_NAMES[top_idx]
        conf = pred[top_idx]
        
        full_text.append(char)
        details.append({
            "char": char,
            "confidence": float(conf),
            "box": sorted_boxes[i]
        })

    return " ".join(full_text), details

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parallel_inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    
    print(f"Processing: {image_path}")
    
    try:
        text, details = recognize_inscription(image_path)
        print("\n--- Recognition Result ---")
        print(f"Full Text: {text}")
        print("\n--- Details ---")
        for i, d in enumerate(details):
            print(f"{i+1}. {d['char']} ({d['confidence']:.2%}) - Box: {d['box']}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
