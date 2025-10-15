<template>
  <div class="ocr-uploader theme-panel">
    <h2>Upload Brahmi Inscription Image</h2>

    <input
      type="file"
      ref="fileInput"
      @change="onFileSelected"
      accept="image/*"
      style="display: none;"
      id="file-upload-input"
    />
    
    <input
      type="file"
      ref="cameraInput"
      @change="onFileSelected"
      accept="image/*"
      capture="environment" 
      style="display: none;"
      id="camera-input"
    />

    <div class="controls upload-buttons">
      <button @click="openFilePicker" :disabled="isLoadingModel || isLoading" class="icon-button">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="24" height="24">
          <path d="M15 17L12 14L9 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M12 14V21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M19 19H20C20.5304 19 21.0391 18.7893 21.4142 18.4142C21.7893 18.0391 22 17.5304 22 17V7C22 6.46957 21.7893 5.96086 21.4142 5.58579C21.0391 5.21071 20.5304 5 20 5H4C3.46957 5 2.96086 5.21071 2.58579 5.58579C2.21071 5.96086 2 6.46957 2 7V17C2 17.5304 2.21071 18.0391 2.58579 18.4142C2.96086 18.7893 3.46957 19 4 19H5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M9 10C10.6569 10 12 8.65685 12 7C12 5.34315 10.6569 4 9 4C7.34315 4 6 5.34315 6 7C6 8.65685 7.34315 10 9 10Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>Upload Inscription</span>
      </button>

      <button @click="openCamera" :disabled="isLoadingModel || isLoading" class="icon-button">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="24" height="24">
          <path d="M23 19C23 19.5304 22.7893 20.0391 22.4142 20.4142C22.0391 20.7893 21.5304 21 21 21H3C2.46957 21 1.96086 20.7893 1.58579 20.4142C1.21071 20.0391 1 19.5304 1 19V8C1 7.46957 1.21071 6.96086 1.58579 6.58579C1.96086 6.21071 2.46957 6 3 6H7L9 3H15L17 6H21C21.5304 6 22.0391 6.21071 22.4142 6.58579C22.7893 6.96086 23 7.46957 23 8V19Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M12 17C14.7614 17 17 14.7614 17 12C17 9.23858 14.7614 7 12 7C9.23858 7 7 9.23858 7 12C7 14.7614 9.23858 17 12 17Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>Capture Photo</span>
      </button>
    </div>

    <p v-if="isLoadingModel" class="loading-message">
      Loading Brahmi OCR Model... Please wait.
    </p>

    <div v-if="localImageUrl" class="image-preview">
      <h3>Selected Image Preview:</h3>
      <img ref="imgRef" :src="localImageUrl" alt="Selected Inscription" class="preview-img theme-border" />
      <button 
        @click="runOcrAndDisplay" 
        :disabled="isLoading || isLoadingModel" 
        class="run-ocr-button"
      >
        {{ isLoading ? 'Recognizing...' : 'Run Brahmi OCR' }}
      </button>
    </div>

    <p v-if="error" class="error-message">{{ error }}</p>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import * as tf from '@tensorflow/tfjs';

// --- CONFIGURATION ---
const IMAGE_SIZE = 64;
const MODEL_PATH = '/brahmi_model/model.json'

// CRITICAL: Replace this with your actual 287 class names
const CLASS_NAMES = [
  'A', 'AA', 'I', 'II', 'U', 'UU', 'E', 'AI', 'O', 'AU',
  'KA', 'KHA', 'GA', 'GHAA', 'NGA', 'CHA', 'CHHA', 'JA', 'JHAA', 'NYA',
  ...Array.from({ length: 267 }, (_, i) => `CHAR_${i + 1}`),
];

// State
const fileInput = ref(null);
const cameraInput = ref(null);
const selectedFile = ref(null);
const localImageUrl = ref('');
const isLoading = ref(false);
const isLoadingModel = ref(true);
const error = ref(null);
const model = ref(null);
const imgRef = ref(null);

const emit = defineEmits(['ocr-result']);

// Helper: Convert File to Base64
const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64String = reader.result.split(',')[1];
      resolve(base64String);
    };
    reader.onerror = error => reject(error);
    reader.readAsDataURL(file);
  });
};

// Run prediction
const runPrediction = async (imageElement) => {
  let tensor = tf.browser.fromPixels(imageElement, 3) // Force 3 channels
    .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
    .toFloat()
    .div(tf.scalar(255.0))
    .expandDims();

  const predictions = model.value.predict(tensor);
  const values = predictions.dataSync();
  const arr = Array.from(values);
  const predictedIndex = arr.indexOf(Math.max(...arr));
  const predictedChar = CLASS_NAMES[predictedIndex];
  const confidence = arr[predictedIndex] * 100;

  tensor.dispose();
  predictions.dispose();

  return { predictedChar, confidence };
};

const runOcrAndDisplay = async () => {
  if (!selectedFile.value || !model.value) {
    error.value = "Model not loaded or image not selected.";
    return;
  }
  
  isLoading.value = true;
  error.value = null;

  try {
    const imageBase64 = await fileToBase64(selectedFile.value);
    const { predictedChar, confidence } = await runPrediction(imgRef.value);

    const result = {
      original_image_b64: imageBase64,
      restored_image_b64: imageBase64,
      recognized_brahmi_ascii: `${predictedChar} (Confidence: ${confidence.toFixed(2)}%)\n\n[Placeholder for full Inscription Recognition]`,
      latin_transliteration: `Prediction: ${predictedChar}. Full sentence translation logic required.\n\n[Placeholder for Latin Transliteration]`,
      devnagri_transliteration: `अनुमान: ${predictedChar}। पूरे वाक्य के अनुवाद तर्क की आवश्यकता है।\n\n[देवनागरी लिप्यंतरण के लिए प्लेसहोल्डर]`,
    };

    emit('ocr-result', result);
  } catch (err) {
    console.error('OCR Prediction Error:', err);
    error.value = 'An error occurred during OCR prediction.';
  } finally {
    isLoading.value = false;
  }
};

// Load model with proper error handling
onMounted(async () => {
  try {
    console.log(`Attempting to load model from: ${MODEL_PATH}`);
    
    // Ensure the TF.js backend is ready before any operations
    await tf.ready();
    console.log('TensorFlow.js backend ready.');

    // Try to load the model
    model.value = await tf.loadLayersModel(MODEL_PATH);
    
    isLoadingModel.value = false;
    console.log('Brahmi OCR Model loaded successfully!');
    console.log('Model input shape:', model.value.inputs[0].shape);
  } catch (e) {
    console.error('TF.js Model Loading Error:', e);
    
    // Provide more specific error messages
    if (e.message.includes('InputLayer')) {
      error.value = 'Model format error: The model file needs to be reconverted with proper input shape configuration. Please check the model conversion process.';
    } else if (e.message.includes('fetch')) {
      error.value = 'Failed to load model files. Please ensure the model files are in the /public/brahmi_model/ folder.';
    } else {
      error.value = `Failed to load OCR model: ${e.message}`;
    }
    
    isLoadingModel.value = false;
  }
});

const openFilePicker = () => {
  fileInput.value.value = '';
  cameraInput.value.value = '';
  fileInput.value.click();
};

const openCamera = () => {
  fileInput.value.value = '';
  cameraInput.value.value = '';
  cameraInput.value.click();
};

const onFileSelected = (event) => {
  error.value = null;
  selectedFile.value = event.target.files[0];
  
  if (selectedFile.value) {
    if (localImageUrl.value) URL.revokeObjectURL(localImageUrl.value);
    localImageUrl.value = URL.createObjectURL(selectedFile.value);
  } else {
    localImageUrl.value = '';
  }
};
</script>

<style scoped>
.ocr-uploader {
  text-align: center;
}

.theme-panel {
  padding: 25px;
  background-color: var(--color-paper-sepia);
  border: 4px solid var(--color-border-clay);
  border-radius: 8px;
  box-shadow: 6px 6px 12px rgba(0, 0, 0, 0.1);
}

.upload-buttons {
  display: flex;
  gap: 15px;
  justify-content: center;
  margin: 20px 0;
}

.icon-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-width: 180px;
}

.icon-button svg {
  stroke: #fff;
  width: 20px;
  height: 20px;
}

.image-preview {
  margin-top: 25px;
  padding-top: 20px;
  border-top: 1px dashed var(--color-border-clay);
}

.preview-img {
  max-width: 100%;
  max-height: 350px;
  display: block;
  margin: 20px auto;
  padding: 5px;
  background-color: #fff;
}

.theme-border {
  border: 2px solid var(--color-border-clay);
  box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.run-ocr-button {
  margin-top: 15px;
  font-weight: 500;
  padding: 12px 25px;
  background-color: var(--color-highlight-gold);
  border-color: var(--color-highlight-gold);
}

.run-ocr-button:hover:not(:disabled) {
  background-color: var(--color-button-hover);
  border-color: var(--color-button-hover);
}

.error-message {
  color: var(--color-error);
  font-weight: 500;
  margin-top: 15px;
  background-color: #ffe0e0;
  padding: 10px;
  border-radius: 4px;
  border: 1px solid var(--color-error);
}

.loading-message {
  color: var(--color-text-maroon);
  font-weight: 600;
  margin-top: 15px;
}

@media (max-width: 550px) {
  .upload-buttons {
    flex-direction: column;
    align-items: center;
  }
  .icon-button {
    width: 100%;
    min-width: unset;
  }
}
</style>