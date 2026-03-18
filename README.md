# Brahmi OCR & Restoration System

Ancient Brahmi inscriptions are among the earliest written records of the Indian subcontinent and form a crucial part of India's linguistic and cultural heritage. Due to the passage of centuries, these inscriptions have deteriorated, resulting in eroded, partial, or missing characters. Conventional OCR systems fail to interpret such degraded text.

**Brahmi OCR** is a deep-learning-based AI system that digitizes, restores, and transliterates eroded Brahmi inscriptions. The system features a two-stage framework integrating Generative Adversarial Networks (GANs) for character restoration and advanced Convolutional Neural Networks (CNNs) for Character Recognition.

---

## 🚀 Key Features

- **Image Noise Cleaning & Binarization**: Advanced OpenCV pipelines to remove stone noise, salt-and-pepper artifacts, and binarize the inscriptions using Otsu & Adaptive thresholding.
- **Smart Character Segmentation**: Automated bounding box detection with aspect ratio normalization and nested box merging.
- **GAN-Based Character Restoration**: Utilizes a Pix2Pix Generative Adversarial Network to reconstruct eroded and damaged character strokes before recognition.
- **High-Accuracy OCR**: Multi-model support including PyTorch (ResNet50, EfficientNetB0) and ONNX (MobileNetV2) with Ensemble capabilities for highest confidence predictions.
- **Transliteration**: Automatically maps identified Brahmi (Latin/Romanized labels) into modern readable Devanagari script.
- **Interactive Web UI**: Built with Vue 3 and Vite, providing a seamless experience to upload images, view restored crops, and export results (PDF).

---

## 🛠️ Technology Stack

### What is Used:
**Frontend:**
- **Vue.js (v3)**: Core UI framework.
- **Vite**: State-of-the-art, lightning-fast build tool and development server.
- **Axios**: HTTP client for API communication.
- **jsPDF**: For exporting recognized texts and reports.
- **TensorFlow.js**: Used in frontend utilities.

**Backend:**
- **Python 3.8+**: Core programming language.
- **Flask**: Lightweight WSGI web application framework serving the REST API.
- **PyTorch & torchvision**: For running the primary ResNet50, EfficientNetB0, and GAN Restorer models.
- **ONNXRuntime**: For high-performance inference of the MobileNetV2 model.
- **TensorFlow / Keras**: Maintained for legacy `.keras` model support and custom transformer layer loading.
- **OpenCV (cv2) & Pillow (PIL)**: For complex image processing, connected-components analysis, noise removal, and letterbox padding.

### What is NOT Used (Deprecated/Replaced):
- Heavy full-page monolithic OCR pipelines (replaced by crop-based classification).
- Standard Tesseract OCR (fails on ancient eroded scripts; hence custom CNNs are used).
- Pure Keras workflows (migrated towards PyTorch/ONNX for better performance and training control, though Keras serialization is retained for backwards compatibility).

---

## 🧠 System Architecture & Workflow

The system follows a highly modular, multi-step pipeline to process an uploaded image of a Brahmi inscription.

### 1. Preprocessing & Segmentation (`process` phase)
1. **Input**: User uploads a Brahmi inscription image via the Vue frontend.
2. **Noise Cleaning**: The backend converts the image to grayscale, applies Median Blur, Otsu Thresholding, and morphological closing to remove background stone textures and 1px noise.
3. **Segmentation**: Bounding boxes around individual characters are detected using Adaptive Thresholding. Nested boxes (100% overlap) are merged. Boxes are padded and normalized to a standard aspect ratio (0.7145 width/height) mimicking the training dataset.
4. **Sorting**: Characters are sorted top-to-bottom, left-to-right to maintain natural reading order.

### 2. Restoration (`GAN` phase)
1. **Cropping**: Original RGB crops are extracted for each detected bounding box.
2. **Pix2Pix GAN**: Each crop is passed through the `GANRestorer` (if loaded) to reconstruct missing strokes and eroded parts.
3. **Post-GAN Binarization**: The restored crop is cleaned again (binarized to stark black and white text) to ensure the OCR model receives clean features.

### 3. Optical Character Recognition (`predict` phase)
1. **Letterboxing**: Crops are padded to a perfect square and resized to model-specific dimensions (typically 224x224).
2. **Inference**: The processed crops are passed into the selected model:
   - **ResNet50** (PyTorch)
   - **EfficientNetB0** (PyTorch)
   - **MobileNetV2** (ONNX)
   - **Ensemble Mode**: Runs all available models and picks the maximum confidence prediction for *each* individual character.
3. **Temperature Scaling**: ONNX logits are temperature-scaled (T=0.2) to calibrate prediction confidence.

### 4. Transliteration & Output
1. The CNN outputs a class index which is mapped to a Romanized Brahmi label (e.g., "ka", "kha").
2. The `roman_to_devanagari` mapping converts this into Devanagari script (e.g., "क", "ख") using `transliteration_mapping.json`.
3. The API returns the bounding boxes, confidence scores, transliterated text, and a base64 composite image showing the GAN-restored characters back in their original positions.

---

## ⚙️ Installation & Setup

### Prerequisites
- **Node.js**: v20.19.0 or higher
- **Python**: v3.8 to v3.12
- **CUDA Toolkit**: (Optional but recommended) for GPU-accelerated PyTorch/ONNX inference.

### 1. Backend Setup
Navigate to the `backend` directory and install dependencies:
```bash
cd backend
python -m venv venv
# Activate venv:
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

**Model Weights:**
Ensure the following directories and their respective weights exist in `backend/`:
- `brahmi_model_resnet50_new/resnet50_brahmi.pth`
- `brahmi_model_efficientnetb0_new/model_weights.pth`
- `brahmi_model_mobilenet_v2/brahmi_ocr_best.onnx`
- `gan_character_restorer/pix2pix_final_epoch_10.pth`

Start the Flask Server:
```bash
python app.py
```
*The backend will run on `http://127.0.0.1:5000`*

### 2. Frontend Setup
Navigate to the project root:
```bash
npm install
```

Start the Vite Development Server:
```bash
npm run dev
```
*The frontend will run on `http://localhost:5173`. Open this URL in your browser.*

---

## 📡 API Endpoints

The Flask backend exposes the following REST endpoints:

- `GET /health` : Returns system health, loaded models, and configuration status.
- `POST /process` : Accepts an image, performs noise cleaning, segmentation, and GAN restoration without running the classification models. Useful for previewing bounding boxes. Returns base64 images and box coordinates.
- `POST /predict` : Accepts an image, target `model` name, and `transliteration` type. Executes the full pipeline (Clean -> Segment -> Restore -> OCR -> Transliterate) and returns predicted text, confidence scores, and bounding boxes.

---

## 📜 License
This project workflow and associated machine learning models are developed for interpreting and preserving Brahmi scripts.