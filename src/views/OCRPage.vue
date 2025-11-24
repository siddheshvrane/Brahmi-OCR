<template>
  <div class="ocr-page">
    <div class="header-section">
      <img :src="logoImg" alt="Brahmi OCR Logo" class="app-logo" />
      <p class="subtitle">Rediscovering Ancient Wisdom through AI</p>
    </div>
    
    <div class="app-layout">
      <div class="layout-column left-panel">
        <FileUploader @ocr-result="handleOcrResult" />
      </div>

      <div class="layout-column right-panel">
        <BrahmiResult :results="ocrResults" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import FileUploader from '../components/FileUploader.vue';
import BrahmiResult from '../components/BrahmiResult.vue';
import logoImg from '../assets/logo.png';

// State for the OCR results
const ocrResults = ref(null);

const handleOcrResult = (data) => {
  ocrResults.value = data;
};
</script>

<style scoped>
.ocr-page {
  width: 100%;
  height: 100vh;
  min-height: 0; /* Crucial for nested scrolling */
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  display: flex;
  flex-direction: column;
  padding: 20px;
  box-sizing: border-box;
}

.header-section {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 20px;
  flex-shrink: 0;
  padding-left: 10px;
}

.app-logo {
  max-height: 120px;
  width: auto;
  filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
  margin-bottom: 0;
}

.subtitle {
  color: var(--color-royal-maroon);
  font-family: 'Outfit', sans-serif;
  font-weight: 500;
  letter-spacing: 2px;
  margin: 0;
  opacity: 0.9;
  font-size: 1.2rem;
}

.app-layout {
  display: flex;
  gap: 30px;
  flex: 1;
  min-height: 0; /* Crucial for nested scrolling */
  width: 100%;
}

.layout-column {
  flex: 1;
  min-width: 0;
  height: 100%;
  display: flex;
  flex-direction: column;
}

/* Responsive Design */
@media (max-width: 900px) {
  .ocr-page {
    height: auto;
    min-height: 100vh;
    overflow-y: auto;
  }

  .app-layout {
    flex-direction: column;
    height: auto;
  }

  .layout-column {
    height: 600px; /* Fixed height for mobile scrollable areas */
  }
}
</style>