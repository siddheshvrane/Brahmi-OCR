<template>
  <div class="uploader-card kaavi-border-accent">
    <div class="card-header">
      <h2>Initial Processing</h2>
      <p class="subtitle">Upload inscription to begin restoration</p>
    </div>

    <!-- Upload Zone -->
    <div 
      class="upload-zone"
      :class="{ 'is-dragover': isDragOver }"
      @dragover.prevent="isDragOver = true"
      @dragleave.prevent="isDragOver = false"
      @drop.prevent="handleDrop"
      @click="$refs.fileInput.click()"
    >
      <input 
        type="file" 
        ref="fileInput" 
        @change="handleFileUpload" 
        accept="image/*" 
        hidden 
      />
      
      <div v-if="!previewUrl" class="upload-prompt">
        <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
        <p class="prompt-text">Click or drag image to upload</p>
        <p class="prompt-subtext">SVG, PNG, JPG (max. 10MB)</p>
      </div>

      <div v-else class="preview-container">
        <img :src="previewUrl" alt="Preview" class="preview-image" />
        <button @click.stop="resetUpload" class="btn-clear" aria-label="Clear image">
          <svg viewBox="0 0 24 24" fill="none" width="16" height="16" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>

    <!-- Actions -->
    <div class="card-footer">
      <button 
        @click="processImage" 
        :disabled="!selectedFile || isProcessing" 
        class="btn-primary"
      >
        <svg v-if="isProcessing" class="spinner" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-dasharray="32" stroke-dashoffset="0" stroke-linecap="round"></circle>
        </svg>
        <span>{{ isProcessing ? 'Restoring Image...' : 'Restore & Segment' }}</span>
      </button>
      
      <div v-if="error" class="error-banner">
        <svg viewBox="0 0 24 24" fill="none" width="16" height="16" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span>{{ error }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const emit = defineEmits(['image-processed']);

const selectedFile = ref(null);
const previewUrl = ref(null);
const isProcessing = ref(false);
const isDragOver = ref(false);
const error = ref(null);

const API_URL = 'http://localhost:5000';

const handleFileUpload = (event) => {
  const file = event.target.files[0];
  if (file) setupFile(file);
};

const handleDrop = (event) => {
  isDragOver.value = false;
  const file = event.dataTransfer.files[0];
  if (file) setupFile(file);
};

const setupFile = (file) => {
  if (!file.type.startsWith('image/')) {
    error.value = "Please upload an image file.";
    return;
  }
  selectedFile.value = file;
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = URL.createObjectURL(file);
  error.value = null;
};

const resetUpload = () => {
  selectedFile.value = null;
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = null;
  error.value = null;
};

const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = error => reject(error);
    reader.readAsDataURL(file);
  });
};

const processImage = async () => {
  if (!selectedFile.value) return;

  isProcessing.value = true;
  error.value = null;

  try {
    const imageBase64 = await fileToBase64(selectedFile.value);
    
    // Step 1: Fast segmentation only — no GAN yet.
    // User will review/edit the returned boxes before triggering GAN.
    const response = await fetch(`${API_URL}/segment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64 }),
    });

    const data = await response.json();
    if (data.success) {
      emit('image-processed', {
        phase: 'segmentation',              // tells BrahmiResult which step we're in
        original_image_b64: data.original_image_b64,
        restored_image_b64: null,           // no GAN yet
        initial_boxes: data.boxes
      });
    } else {
      error.value = data.error || "Segmentation failed.";
    }
  } catch (err) {
    console.error(err);
    error.value = "Server error. Ensure backend is running.";
  } finally {
    isProcessing.value = false;
  }
};
</script>

<style scoped>
.uploader-card {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  padding: 32px; /* A bit more breathing room */
  display: flex;
  flex-direction: column;
  gap: 28px;
}

/* Header */
.card-header h2 {
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0;
  letter-spacing: -0.01em;
}

.subtitle {
  font-size: 0.875rem;
  color: var(--color-text-secondary);
  margin: 4px 0 0 0;
}

/* Upload Zone */
.upload-zone {
  flex: 1;
  min-height: 200px;
  border: 1px dashed var(--color-border);
  border-radius: var(--radius-md);
  background-color: var(--color-bg-app);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
}

.upload-zone:hover, .upload-zone.is-dragover {
  border-color: var(--color-kaavi-red);
  background-color: rgba(139, 44, 36, 0.02); /* Very subtle kaavi tint */
}

/* Upload Prompt */
.upload-prompt {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 20px;
}

.upload-icon {
  width: 40px;
  height: 40px;
  color: var(--color-text-secondary);
  margin-bottom: 12px;
}

.prompt-text {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--color-text-primary);
  margin: 0;
}

.prompt-subtext {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
  margin: 4px 0 0 0;
}

/* Preview */
.preview-container {
  width: 100%;
  height: 100%;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 12px;
}

.preview-image {
  max-width: 100%;
  max-height: 200px;
  border-radius: 8px;
  object-fit: contain;
}

.btn-clear {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  color: var(--color-text-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow-sm);
}

.btn-clear:hover {
  background-color: var(--color-surface-hover);
  color: var(--color-text-primary);
}

/* Footer & Actions */
.card-footer {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.btn-primary {
  width: 100%;
  background-color: var(--color-kaavi-red);
  color: white;
  border: none;
  border-radius: var(--radius-md);
  padding: 14px 20px;
  font-size: 0.95rem;
  letter-spacing: 0.5px;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  box-shadow: var(--shadow-sm);
}

.btn-primary:hover:not(:disabled) {
  background-color: var(--color-kaavi-earth);
  transform: translateY(-1px);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.spinner {
  width: 16px;
  height: 16px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.error-banner {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background-color: #fef2f2;
  border-radius: 8px;
  color: #b91c1c;
  font-size: 0.875rem;
  font-weight: 500;
}
</style>