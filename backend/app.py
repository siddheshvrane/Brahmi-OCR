"""
Brahmi OCR Backend - Python Flask API
Usage:
  1. Install dependencies: pip install -r requirements.txt
  2. Run: python app.py
  3. Backend will run at: http://127.0.0.1:5000
"""

import sys
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import os
import cv2
import onnxruntime as ort
import torch

# Import segmentation and GAN restorer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segmentation import detect_characters, sort_boxes, clean_image_noise
from gan_restorer import GANRestorer


# ============================================================================
# MODEL LOADING
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    'ResNet50': os.path.join(BASE_DIR, 'brahmi_model_resnet50_new', 'resnet50_brahmi.pth'),
    'EfficientNetB0': os.path.join(BASE_DIR, 'brahmi_model_efficientnetb0_new', 'model_weights.pth'),
    'MobileNetV2': os.path.join(BASE_DIR, 'brahmi_model_mobilenet_v2', 'brahmi_ocr_best.onnx')
}

CONFIG_PATHS = {
    'ResNet50': os.path.join(BASE_DIR, 'brahmi_model_resnet50_new', 'class_names.json'),
    'EfficientNetB0': os.path.join(BASE_DIR, 'brahmi_model_efficientnetb0_new', 'model_config_efficientnetb0_new.json'),
    'MobileNetV2': os.path.join(BASE_DIR, 'brahmi_model_mobilenet_v2', 'brahmi_ocr_best_config.json')
}

models = {}
configs = {}
translit_mapping = {}

# Load configuration for class names and dimensions
for model_name, config_path in CONFIG_PATHS.items():
    try:
        print(f"Loading configuration for {model_name} from {config_path}...")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        if isinstance(config_data, list):
            # Convert list of class names to standard config dict
            configs[model_name] = {
                'class_names': config_data,
                'num_classes': len(config_data),
                'image_height': 224,
                'image_width': 224
            }
        else:
            configs[model_name] = config_data

        cfg = configs[model_name]
        class_names = cfg.get('class_names', [])

        # Unified naming for class mapping
        if 'idx2label' in cfg:
            idx2label = cfg['idx2label']
            max_id = max(int(k) for k in idx2label.keys())
            class_names = [idx2label.get(str(i), '<UNK>') for i in range(max_id + 1)]
            cfg['class_names'] = class_names
        elif 'idx2char' in cfg:
            idx2char = cfg['idx2char']
            max_id = max(int(k) for k in idx2char.keys())
            class_names = [idx2char.get(str(i), '<UNK>') for i in range(max_id + 1)]
            cfg['class_names'] = class_names
        elif 'id2char' in cfg:
            id2char = cfg['id2char']
            max_id = max(int(k) for k in id2char.keys())
            class_names = [id2char.get(str(i), '<UNK>') for i in range(max_id + 1)]
            cfg['class_names'] = class_names

        if not cfg.get('num_classes'):
            cfg['num_classes'] = len(class_names)

        print(f"OK Configuration loaded for {model_name}: {len(class_names)} classes")

    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}.")
    except Exception as e:
        print(f"ERROR loading configuration for {model_name}: {e}")

print("Loading Brahmi OCR models...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize GAN Restorer
gan_restorer = None
try:
    gan_path = os.path.join(BASE_DIR, 'gan_character_restorer', 'pix2pix_final_epoch_10.pth')
    if os.path.exists(gan_path):
        gan_restorer = GANRestorer(gan_path, device=device)
    else:
        print(f"WARNING: GAN model not found at {gan_path}")
except Exception as e:
    print(f"ERROR loading GAN Restorer: {e}")

for model_name, model_path in MODEL_PATHS.items():
    try:
        print(f"Loading {model_name} from {model_path}...")
        if model_path.endswith('.onnx'):
            models[model_name] = ort.InferenceSession(model_path)
            print(f"OK {model_name} ONNX session loaded successfully!")
        elif model_path.endswith('.pth'):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            # Get num_classes from config if not in checkpoint
            num_classes = checkpoint.get("num_classes") or configs.get(model_name, {}).get("num_classes", 214)

            if "resnet50" in model_path.lower():
                from brahmi_model_resnet50_new.model import ResNet50Classifier
                pt_model = ResNet50Classifier(num_classes=num_classes)
            else:
                from brahmi_model_efficientnetb0_new.model import EfficientNetB0Classifier
                pt_model = EfficientNetB0Classifier(num_classes=num_classes)

            pt_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            pt_model.to(device)
            pt_model.eval()

            models[model_name] = pt_model
            print(f"OK {model_name} PyTorch model loaded successfully on {device}!")

    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}.")
    except Exception as e:
        print(f"ERROR loading {model_name}: {e}")
        import traceback
        traceback.print_exc()

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


# ============================================================================
# FLASK API
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests


def resize_with_padding(img, target_width, target_height, padding_percent=0.15):
    """
    Resizes image to target dimensions using 'Padding to Square' method:
    1. Calculate the largest dimension of the crop.
    2. Add extra padding on all sides (15%) to mimic training data constraints.
    3. Pad the original image with a white background to a perfect square.
    4. Resize to target_width x target_height.
    """
    original_w, original_h = img.size
    max_dim = max(original_w, original_h)

    padding = int(max_dim * padding_percent)
    new_dim = max_dim + 2 * padding

    square_img = Image.new("RGB", (new_dim, new_dim), (255, 255, 255))
    offset_x = (new_dim - original_w) // 2
    offset_y = (new_dim - original_h) // 2
    square_img.paste(img, (offset_x, offset_y))

    return square_img.resize((target_width, target_height), Image.Resampling.LANCZOS)


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

        if img.mode != 'RGB':
            img = img.convert('RGB')

        # --- 2. Determine Request Params ---
        model_name = request.form.get('model') or (request.json and request.json.get('model')) or 'ResNet50'
        transliteration = request.form.get('transliteration') or (request.json and request.json.get('transliteration')) or 'latin'

        # Keep original PIL Image for GAN crops
        original_pil = img.copy()

        # --- 3. Segmentation ---
        open_cv_image = np.array(img)[:, :, ::-1].copy()  # RGB -> BGR

        # Use cleaned image ONLY for finding bounding boxes
        detection_image = clean_image_noise(open_cv_image, min_dot_area=50)

        # Get custom boxes from request if they exist
        custom_boxes = (request.json and request.json.get('boxes')) or request.form.get('boxes')

        if custom_boxes:
            if isinstance(custom_boxes, str):
                sorted_boxes = json.loads(custom_boxes)
            else:
                sorted_boxes = custom_boxes
            print(f"Using {len(sorted_boxes)} custom/manual boxes.")
        else:
            boxes, _ = detect_characters(detection_image)

            if len(boxes) <= 1:
                sorted_boxes = [[0, 0, open_cv_image.shape[1], open_cv_image.shape[0]]]
            else:
                sorted_boxes = sort_boxes(boxes)

        # --- GAN Restoration & Then Cleaning (Per Crop) ---
        restored_pil_crops = []
        if sorted_boxes:
            composite_img = original_pil.copy()

            for (x, y, w, h) in sorted_boxes:
                # 1. Crop from original image
                crop = original_pil.crop((x, y, x+w, y+h))

                # 2. GAN Restore
                restored_crop = gan_restorer.restore(crop) if gan_restorer else crop

                # 3. Clean noise & binarize AFTER GAN
                restored_cv = np.array(restored_crop)[:, :, ::-1].copy()
                cleaned_restored_cv = clean_image_noise(restored_cv, min_dot_area=10)
                cleaned_restored_pil = Image.fromarray(cleaned_restored_cv[:, :, ::-1])

                # Resize cleaned result back to original box size for composite display
                composite_img.paste(cleaned_restored_pil.resize((w, h), Image.Resampling.LANCZOS), (x, y))

                restored_pil_crops.append(cleaned_restored_pil)

            restored_image_to_show = composite_img
        else:
            restored_image_to_show = original_pil

        # Encode the restored image to return to the frontend
        buffered = io.BytesIO()
        restored_image_to_show.save(buffered, format="JPEG")
        restored_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # --- 4. Prediction Logic ---
        results = []

        # ImageNet normalization constants
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def get_model_preds(model_key, crop_images):
            model_obj = models[model_key]
            cfg = configs.get(model_key, {})
            is_onnx = MODEL_PATHS.get(model_key, "").endswith('.onnx')

            h = cfg.get('image_height', 224)
            w = cfg.get('image_width', 224)

            batch_arr = []
            for c_img in crop_images:
                c_arr = np.array(resize_with_padding(c_img, w, h), dtype=np.float32)

                if is_onnx:
                    # ONNX MobileNetV2: CHW layout + ImageNet normalization
                    c_arr = c_arr / 255.0
                    c_arr = (c_arr - IMAGENET_MEAN) / IMAGENET_STD
                    c_arr = np.transpose(c_arr, (2, 0, 1))  # HWC -> CHW
                else:
                    # PyTorch models: normalize to [0, 1] first (mean/std applied below)
                    c_arr = c_arr / 255.0

                batch_arr.append(c_arr)

            batch_input = np.array(batch_arr)

            if is_onnx:
                logits = model_obj.run(None, {'image': batch_input})[0]
                # Temperature scaling (T=0.2) to sharpen confidence — ONNX model was
                # trained with label_smoothing=0.1 causing small raw logit range.
                return logits / 0.2
            else:
                # PyTorch model: apply ImageNet normalization in NCHW layout
                batch_tensor = torch.from_numpy(batch_input).permute(0, 3, 1, 2).to(device)
                mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
                std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
                batch_tensor = (batch_tensor - mean) / std
                with torch.no_grad():
                    logits = model_obj(batch_tensor)
                return logits.cpu().numpy()

        pil_crops = restored_pil_crops

        # Resolve class names
        label_model = model_name if model_name in configs else (list(configs.keys())[0] if configs else None)
        if not label_model:
            return jsonify({'success': False, 'error': 'No model configurations loaded.'}), 500
        class_names = configs[label_model].get('class_names', [])

        if model_name == 'Ensemble':
            if not models:
                return jsonify({'success': False, 'error': 'No models for Ensemble.'}), 500

            num_crops = len(pil_crops)
            num_classes = len(class_names)
            all_model_probs = []
            ensemble_errors = {}

            print(f"Ensemble (Per-Char Max-Conf): Processing {num_crops} chars with {len(models)} models...")

            for m_key in models.keys():
                try:
                    preds = get_model_preds(m_key, pil_crops)
                    probs = torch.nn.functional.softmax(torch.from_numpy(preds), dim=-1).numpy()
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

            if not all_model_probs:
                return jsonify({'success': False, 'error': 'Ensemble failed completely.', 'details': ensemble_errors}), 500

            # Per-character maximum confidence selection
            final_probabilities = np.zeros((num_crops, num_classes))
            for i in range(num_crops):
                best_m_idx = max(range(len(all_model_probs)), key=lambda m: np.max(all_model_probs[m][i]))
                final_probabilities[i] = all_model_probs[best_m_idx][i]

        elif model_name in models:
            preds = get_model_preds(model_name, pil_crops)
            final_probabilities = torch.nn.functional.softmax(torch.from_numpy(preds), dim=-1).numpy()
        else:
            return jsonify({'success': False, 'error': f"Model '{model_name}' not found."}), 400

        # --- 5. Decode Results ---
        full_text_latin = []
        full_text_devanagari = []
        CONF_THRESHOLD = 20.0  # Drop boxes with < 20% confidence

        for i, probs in enumerate(final_probabilities):
            top_idx = np.argmax(probs)
            conf = float(probs[top_idx] * 100)

            if conf < CONF_THRESHOLD:
                print(f"Dropping low-confidence box {i} ({conf:.2f}%)")
                continue

            char_name_latin = class_names[top_idx] if top_idx < len(class_names) else "Unknown"
            char_name_devanagari = roman_to_devanagari(char_name_latin)

            full_text_latin.append(char_name_latin)
            full_text_devanagari.append(char_name_devanagari)
            results.append({
                'character': char_name_latin,
                'character_devanagari': char_name_devanagari,
                'confidence': conf,
                'box': sorted_boxes[i]
            })

        top_conf = sum(r['confidence'] for r in results) / len(results) if results else 0.0

        return jsonify({
            'success': True,
            'top_prediction': " ".join(full_text_latin),
            'top_prediction_devanagari': " ".join(full_text_devanagari),
            'top_confidence': top_conf,
            'predictions': results,
            'model_used': model_name,
            'restored_image_b64': restored_image_b64
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f"Internal Error: {str(e)}"}), 500


@app.route('/process', methods=['POST'])
def process():
    """Initial image processing and character segmentation without prediction."""
    try:
        # Load Image
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream)
        elif request.json and 'image' in request.json:
            img_data = base64.b64decode(request.json['image'])
            img = Image.open(io.BytesIO(img_data))
        else:
            return jsonify({'success': False, 'error': 'No image provided.'}), 400

        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Encode original image for frontend
        buffered_orig = io.BytesIO()
        img.save(buffered_orig, format="JPEG")
        original_image_b64 = base64.b64encode(buffered_orig.getvalue()).decode('utf-8')

        # Detect characters on a cleaned version
        open_cv_image = np.array(img)[:, :, ::-1].copy()
        detection_image = clean_image_noise(open_cv_image, min_dot_area=50)
        boxes, _ = detect_characters(detection_image)
        sorted_boxes = sort_boxes(boxes) if boxes else []

        # Build GAN composite: original crop → GAN restore → clean & binarize → paste back
        if sorted_boxes:
            composite_img = img.copy()
            for (x, y, w, h) in sorted_boxes:
                crop = img.crop((x, y, x+w, y+h))

                restored_crop = gan_restorer.restore(crop) if gan_restorer else crop

                restored_cv = np.array(restored_crop)[:, :, ::-1].copy()
                cleaned_restored_cv = clean_image_noise(restored_cv, min_dot_area=10)
                cleaned_restored_pil = Image.fromarray(cleaned_restored_cv[:, :, ::-1])

                composite_img.paste(cleaned_restored_pil.resize((w, h), Image.Resampling.LANCZOS), (x, y))
            display_pil = composite_img
        else:
            display_pil = img

        # Encode processed image for frontend
        buffered_clean = io.BytesIO()
        display_pil.save(buffered_clean, format="JPEG")
        restored_image_b64 = base64.b64encode(buffered_clean.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'restored_image_b64': restored_image_b64,
            'original_image_b64': original_image_b64,
            'boxes': sorted_boxes
        })

    except Exception as e:
        print(f"Processing error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
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
            '/predict': 'POST - Predict character from image',
            '/process': 'POST - Segment image without prediction'
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Brahmi OCR Backend Server Starting...")
    print("="*60)
    print(f"Server running at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)