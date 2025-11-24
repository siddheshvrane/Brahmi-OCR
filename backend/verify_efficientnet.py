import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

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

    def call(self, x):
        max_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        position_embeddings = self.position_embedding(positions)
        return x + position_embeddings

# Path to the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'brahmi_model_efficientnetb0', 'brahmi_model_efficientnetb0.keras')

print(f"Checking model at: {model_path}")

if not os.path.exists(model_path):
    print("ERROR: Model file not found!")
    exit(1)

try:
    custom_objects = {
        'TransformerBlock': TransformerBlock,
        'TokenAndPositionEmbedding': TokenAndPositionEmbedding
    }
    # Use keras.models.load_model as in inference_example.py
    model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    print("SUCCESS: Model loaded successfully!")
except Exception as e:
    print(f"ERROR: Failed to load model. {e}")
    import traceback
    traceback.print_exc()
    exit(1)
