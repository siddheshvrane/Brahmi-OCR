//src/views/OCRPage.vue
<template>
  <div class="ocr-page">
    <div class="title-container">
      <img src="../assets/logo.png" alt="Brahmi Inscription OCR" class="logo-title" />
    </div>
    
    <div class="app-layout">
      <FileUploader @ocr-result="handleOcrResult" class="uploader-panel" />

      <!-- Added check for ocrResults before passing to ensure initial load is clean -->
      <BrahmiResult :results="ocrResults" class="results-panel" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import FileUploader from '../components/FileUploader.vue';
import BrahmiResult from '../components/BrahmiResult.vue';

// State for the OCR results
const ocrResults = ref(null);

const handleOcrResult = (data) => {
  // Updated expected Data structure: 
  // { 
  //   restored_image_b64: '...', 
  //   original_image_b64: '...', 
  //   recognized_brahmi_ascii: '...', 
  //   latin_transliteration: '...',
  //   devnagri_transliteration: '...' 
  // }
  ocrResults.value = data;
};
</script>

<style scoped>
.ocr-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}
.title-container {
  text-align: center;
  margin-bottom: 30px;
}
.logo-title {
  max-width: 500px; /* Adjust as needed */
  height: auto;
}
.app-layout {
  display: flex;
  flex-direction: column; /* Stack divs one below the other */
  gap: 30px;
  margin-top: 30px;
}
.uploader-panel, .results-panel {
  flex: 1;
  min-width: 0;
}
</style>