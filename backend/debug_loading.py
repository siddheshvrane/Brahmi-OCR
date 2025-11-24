import os
import tensorflow as tf
from tensorflow.keras import layers
import json

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

# Dictionary of custom objects needed for Keras to deserialize the model
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding
}

models = {}

print("Loading Brahmi OCR models...")

for model_name, model_path in MODEL_PATHS.items():
    try:
        print(f"Loading {model_name} from {model_path}...")
        
        if not os.path.exists(model_path):
            print(f"ERROR: File does not exist: {model_path}")
            continue

        # Load the .keras model with custom objects
        models[model_name] = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False  # Faster loading for inference only
        )
        print(f"✓ {model_name} loaded successfully!")
        
    except Exception as e:
        print(f"ERROR loading {model_name}: {e}")
        import traceback
        traceback.print_exc()

print(f"\nTotal models loaded: {len(models)}")
