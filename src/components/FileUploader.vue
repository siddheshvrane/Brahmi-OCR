<template>
  <div class="ocr-uploader glass-panel">
    <div class="panel-header">
      <h2>Upload Inscription</h2>
      <div class="model-selector">
        <label for="model-select">OCR Model:</label>
        <select id="model-select" v-model="selectedModel" class="glass-select">
          <option value="ResNet50">ResNet50</option>
          <option value="EfficientNetB0">EfficientNetB0</option>
          <option value="MobileNetV2">MobileNetV2</option>
          <option value="Ensemble">Ensemble (All Models)</option>
        </select>
      </div>
    </div>

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

    <div class="upload-area" @click="openFilePicker" @dragover.prevent @drop.prevent="onFileDrop">
      <div v-if="!localImageUrl" class="upload-placeholder">
        <div class="icon-circle">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 16.5V7.5M12 7.5L8 11.5M12 7.5L16 11.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M20 16.5V18.5C20 19.6046 19.1046 20.5 18 20.5H6C4.89543 20.5 4 19.6046 4 18.5V16.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <p>Click or Drag & Drop to Upload</p>
        <span class="sub-text">Supports JPG, PNG, WEBP</span>
      </div>
      
      <div v-else class="image-preview-container">
        <img :src="localImageUrl" alt="Selected Inscription" class="preview-img" />
        <button class="remove-btn" @click.stop="clearSelection">×</button>
      </div>
    </div>

    <div class="controls">
      <button @click="openCamera" :disabled="isLoading" class="glass-btn secondary">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="20" height="20">
          <path d="M23 19C23 19.5304 22.7893 20.0391 22.4142 20.4142C22.0391 20.7893 21.5304 21 21 21H3C2.46957 21 1.96086 20.7893 1.58579 20.4142C1.21071 20.0391 1 19.5304 1 19V8C1 7.46957 1.21071 6.96086 1.58579 6.58579C1.96086 6.21071 2.46957 6 3 6H7L9 3H15L17 6H21C21.5304 6 22.0391 6.21071 22.4142 6.58579C22.7893 6.96086 23 7.46957 23 8V19Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M12 17C14.7614 17 17 14.7614 17 12C17 9.23858 14.7614 7 12 7C9.23858 7 7 9.23858 7 12C7 14.7614 9.23858 17 12 17Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Camera
      </button>

      <button 
        @click="runOcrAndDisplay" 
        :disabled="isLoading || !selectedFile || !backendReady" 
        class="glass-btn primary"
      >
        <span v-if="isLoading" class="loader"></span>
        {{ isLoading ? 'Deciphering...' : 'Decipher Inscription' }}
      </button>
    </div>

    <div v-if="backendStatus && !backendReady" class="status-message error">
      {{ backendStatus.message }}
    </div>

    <p v-if="error" class="error-message">{{ error }}</p>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

// --- CONFIGURATION ---
const API_URL = 'http://localhost:5000';

// State
const fileInput = ref(null);
const cameraInput = ref(null);
const selectedFile = ref(null);
const localImageUrl = ref('');
const isLoading = ref(false);
const error = ref(null);
const backendReady = ref(false);
const backendStatus = ref(null);
const selectedModel = ref('ResNet50'); // Default model

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

// Check backend health
const checkBackendHealth = async () => {
  try {
    const response = await fetch(`${API_URL}/health`);
    const data = await response.json();
    
    if (data.status === 'healthy') {
      backendReady.value = true;
      backendStatus.value = {
        type: 'success',
        message: `✓ Backend connected`
      };
      return true;
    }
    return false;
  } catch (err) {
    backendReady.value = false;
    backendStatus.value = {
      type: 'error',
      message: '✗ Backend disconnected'
    };
    return false;
  }
};

// Run OCR prediction via backend API
const runOcrAndDisplay = async () => {
  if (!selectedFile.value) {
    error.value = "Please select an image first.";
    return;
  }

  if (!backendReady.value) {
    error.value = "Backend server is not running.";
    return;
  }
  
  isLoading.value = true;
  error.value = null;

  try {
    // Convert image to base64
    const imageBase64 = await fileToBase64(selectedFile.value);

    // Send to backend API
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        image: imageBase64,
        model: selectedModel.value 
      })
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(data.error || 'Unknown error');
    }

    // Format predictions for display
    const topChar = data.top_prediction;
    const topConfidence = data.top_confidence.toFixed(2);
    
    const allPredictions = data.predictions
      .map(p => `${p.character} (${p.confidence.toFixed(2)}%)`)
      .join('\n');

    // Create result object
    const result = {
      original_image_b64: imageBase64,
      restored_image_b64: imageBase64,
      recognized_brahmi_ascii: `Top Prediction: ${topChar}\nConfidence: ${topConfidence}%\n\nAll Predictions:\n${allPredictions}`,
      latin_transliteration: `Character: ${topChar}`,
      devnagri_transliteration: `वर्ण: ${topChar}`,
      model_used: selectedModel.value
    };

    emit('ocr-result', result);
  } catch (err) {
    console.error('OCR Prediction Error:', err);
    error.value = `Error: ${err.message}`;
  } finally {
    isLoading.value = false;
  }
};

// Check backend on mount
onMounted(async () => {
  await checkBackendHealth();
  
  // Recheck every 10 seconds if not connected
  if (!backendReady.value) {
    setInterval(async () => {
      if (!backendReady.value) {
        await checkBackendHealth();
      }
    }, 10000);
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
  const file = event.target.files[0];
  handleFile(file);
};

const onFileDrop = (event) => {
  error.value = null;
  const file = event.dataTransfer.files[0];
  handleFile(file);
};

const handleFile = (file) => {
  if (file && file.type.startsWith('image/')) {
    selectedFile.value = file;
    if (localImageUrl.value) URL.revokeObjectURL(localImageUrl.value);
    localImageUrl.value = URL.createObjectURL(file);
  } else if (file) {
    error.value = "Please upload an image file.";
  }
};

const clearSelection = () => {
  selectedFile.value = null;
  if (localImageUrl.value) URL.revokeObjectURL(localImageUrl.value);
  localImageUrl.value = '';
};
</script>

<style scoped>
.glass-panel {
  background: var(--color-glass-bg);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--color-glass-border);
  border-radius: 16px;
  padding: 24px;
  box-shadow: var(--shadow-glass);
  display: flex;
  flex-direction: column;
  height: 100%;
  box-sizing: border-box;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

h2 {
  font-size: 1.5rem;
  color: var(--color-royal-maroon);
  font-family: 'Cinzel Decorative', cursive;
  text-shadow: none;
  font-weight: 700;
}

.model-selector {
  display: flex;
  align-items: center;
  gap: 10px;
}

.model-selector label {
  color: var(--color-text-secondary);
  font-weight: 600;
}

.glass-select {
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid var(--color-gold-accent);
  color: var(--color-royal-maroon);
  padding: 10px 16px;
  border-radius: 12px;
  outline: none;
  cursor: pointer;
  font-family: 'Outfit', sans-serif;
  font-weight: 600;
  font-size: 1rem;
  backdrop-filter: blur(4px);
  transition: all 0.3s ease;
  appearance: none; /* Remove default arrow */
  background-image: url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%23800000%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E");
  background-repeat: no-repeat;
  background-position: right 12px top 50%;
  background-size: 12px auto;
  padding-right: 40px;
}

.glass-select:hover, .glass-select:focus {
  background-color: rgba(255, 255, 255, 0.25);
  border-color: var(--color-deep-saffron);
  box-shadow: 0 0 15px rgba(212, 175, 55, 0.2);
}

.glass-select option {
  background: var(--color-parchment);
  color: var(--color-royal-maroon);
  font-weight: 500;
  padding: 10px;
}

.upload-area {
  flex: 1;
  border: 2px dashed var(--color-gold-accent);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: rgba(255, 255, 255, 0.2);
  position: relative;
  overflow: hidden;
  min-height: 200px;
}

.upload-area:hover {
  border-color: var(--color-deep-saffron);
  background: rgba(255, 255, 255, 0.3);
}

.upload-placeholder {
  text-align: center;
  color: var(--color-text-secondary);
}

.icon-circle {
  width: 64px;
  height: 64px;
  background: rgba(212, 175, 55, 0.1);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 16px;
  color: var(--color-deep-saffron);
}

.icon-circle svg {
  width: 32px;
  height: 32px;
}

.sub-text {
  display: block;
  font-size: 0.8rem;
  opacity: 0.8;
  margin-top: 8px;
  color: var(--color-text-secondary);
}

.image-preview-container {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  padding: 10px;
}

.preview-img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.remove-btn {
  position: absolute;
  top: 10px;
  right: 10px;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: var(--color-royal-maroon);
  color: var(--color-antique-cream);
  border: none;
  font-size: 1.2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.2s;
}

.remove-btn:hover {
  transform: scale(1.1);
  background: #600000;
}

.controls {
  display: flex;
  gap: 16px;
  margin-top: 20px;
}

.glass-btn {
  flex: 1;
  padding: 12px;
  border-radius: 8px;
  border: none;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  cursor: pointer;
  transition: all 0.3s;
}

.glass-btn.primary {
  background: linear-gradient(45deg, var(--color-royal-maroon), #700000);
  color: var(--color-antique-cream);
  box-shadow: 0 4px 15px rgba(74, 4, 4, 0.3);
}

.glass-btn.primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(74, 4, 4, 0.4);
}

.glass-btn.secondary {
  background: rgba(255, 255, 255, 0.4);
  color: var(--color-royal-maroon);
  border: 1px solid var(--color-gold-accent);
}

.glass-btn.secondary:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.6);
}

.glass-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.status-message {
  margin-top: 10px;
  padding: 10px;
  border-radius: 8px;
  font-size: 0.9rem;
  text-align: center;
  font-weight: 500;
}

.status-message.error {
  background: rgba(255, 0, 0, 0.1);
  color: #800000;
  border: 1px solid rgba(255, 0, 0, 0.2);
}

.error-message {
  color: #c00;
  font-size: 0.9rem;
  margin-top: 10px;
  text-align: center;
}

.loader {
  width: 16px;
  height: 16px;
  border: 2px solid var(--color-antique-cream);
  border-bottom-color: transparent;
  border-radius: 50%;
  display: inline-block;
  box-sizing: border-box;
  animation: rotation 1s linear infinite;
}

@keyframes rotation {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>