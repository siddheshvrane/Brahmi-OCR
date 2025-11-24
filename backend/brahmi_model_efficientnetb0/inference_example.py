#!/usr/bin/env python3
"""
Brahmi OCR Inference Example
Usage: python inference_example.py image.png
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import sys
from PIL import Image

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
        # Escaping curly braces for dictionary literal
        config.update({"embed_dim": self.embed_dim,
                       "num_heads": self.num_heads,
                       "ff_dim": self.ff_dim,
                       "rate": self.rate,})
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
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config()
        # Escaping curly braces for dictionary literal
        config.update({"max_len": self.max_len,
                       "embed_dim": self.embed_dim,})
        return config

    def call(self, inputs, **kwargs):
        # Cast to float32 to avoid type mismatch with position embeddings
        inputs = tf.cast(inputs, tf.float32)
        max_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        position_embeddings = self.position_embedding(positions)
        return inputs + position_embeddings


# ============================================================================
# LOAD MODEL AND CONFIG
# ============================================================================

# Load configuration
with open('model_config_efficientnetb0.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Load class names
with open('class_names_efficientnetb0.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)

# Load model with custom objects
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding
}

model = keras.models.load_model(
    'brahmi_model_efficientnetb0.keras',  # Model filename will be inserted here
    custom_objects=custom_objects,
    compile=False
)

print("✓ Model loaded successfully")
print("✓ Number of classes: {}".format(len(class_names)))


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def predict_image(image_path, top_k=5):
    """
    Predict character in image

    Args:
        image_path: Path to input image
        top_k: Number of top predictions to return

    Returns:
        predictions: List of (character, confidence) tuples
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((config['image_width'], config['image_height']),
                     Image.Resampling.LANCZOS)

    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.clip(img_array, 0.0, 1.0)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]

    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        char = class_names[idx]
        confidence = predictions[idx]
        results.append((char, confidence))

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_example.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Processing: {}".format(image_path))
    print("-" * 50)

    predictions = predict_image(image_path, top_k=5)

    print("Top 5 Predictions:")
    print("-" * 50)
    for i, (char, conf) in enumerate(predictions, 1):
        print("{}. {:20s} : {:6.2f}%".format(i, char, conf*100))

    print("-" * 50)
    print("Predicted Character: {}".format(predictions[0][0]))
    print("Confidence: {:.2f}%".format(predictions[0][1]*100))
