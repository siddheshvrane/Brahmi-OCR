<template>
  <div class="result-display theme-panel">
    <h2>OCR Results</h2>

    <div v-if="!results" class="placeholder">
      <p>Upload an image to begin the Brahmi OCR process.</p>
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" width="48" height="48">
        <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="var(--color-border-clay)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M14 2V8H20" stroke="var(--color-border-clay)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M16 13H8" stroke="var(--color-border-clay)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M16 17H8" stroke="var(--color-border-clay)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M10 9H8" stroke="var(--color-border-clay)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>

    <div v-else class="results-container">
      <h3 class="result-title">Original Inscription Image:</h3>
      <img :src="originalImageSrc" alt="Original Inscription" class="result-img theme-border" />

      <hr />

      <h3 class="result-title">Restored Inscription Image:</h3>
      <img :src="restoredImageSrc" alt="Restored Inscription" class="result-img theme-border" />
      
      <hr />

      <h3 class="result-title">Recognized Brahmi Script (Brahmi ASCII):</h3>
      <div class="recognized-text brahmi-script-display">
        <p>{{ results.recognized_brahmi_ascii }}</p>
      </div>

      <hr />

      <h3 class="result-title">Transliteration:</h3>
      <div class="transliteration-controls">
        <button 
          :class="['lang-button', { active: isLatin }]" 
          @click="isLatin = true"
        >
          Latin Script (Default)
        </button>
        <button 
          :class="['lang-button', { active: !isLatin }]" 
          @click="isLatin = false"
        >
          Devnagri Script
        </button>
      </div>

      <div class="recognized-text transliteration-display" :class="{ 'devnagri-font': !isLatin }">
        <p v-if="isLatin">{{ results.latin_transliteration }}</p>
        <p v-else>{{ results.devnagri_transliteration }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const props = defineProps({
  results: Object, // Expects the object from the API response
});

// Reactive state for the transliteration language
const isLatin = ref(true);

// Converts the Base64 string for the restored image
const restoredImageSrc = computed(() => {
  if (props.results && props.results.restored_image_b64) {
    return `data:image/jpeg;base64,${props.results.restored_image_b64}`;
  }
  return '';
});

// Converts the Base64 string for the original image (NEW)
const originalImageSrc = computed(() => {
  if (props.results && props.results.original_image_b64) {
    return `data:image/jpeg;base64,${props.results.original_image_b64}`;
  }
  return '';
});
</script>

<style scoped>
/* Existing Styles */
.theme-panel {
  padding: 25px;
  background-color: var(--color-paper-sepia);
  border: 4px solid var(--color-border-clay);
  border-radius: 8px;
  box-shadow: 6px 6px 12px rgba(0, 0, 0, 0.1);
}

.placeholder {
  min-height: 250px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: #e3d4bb;
  border-radius: 4px;
  padding: 20px;
  text-align: center;
  color: var(--color-button-hover);
}

.placeholder svg {
  margin-top: 15px;
  stroke-width: 1.5;
}

.result-title {
  border-bottom: 1px solid var(--color-highlight-gold);
  padding-bottom: 5px;
  margin-top: 20px;
}

.result-img {
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

.recognized-text {
  background-color: #fffaf0;
  padding: 20px;
  border-radius: 4px;
  border: 1px dashed var(--color-border-clay);
  font-size: 1.1rem;
  white-space: pre-wrap;
  text-align: left;
  min-height: 100px;
  box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.05);
}

hr {
    border: none;
    height: 1px;
    background-color: var(--color-border-clay);
    margin: 30px 0;
}

/* NEW Styles for Brahmi and Transliteration */
.brahmi-script-display {
  /* You may need to link a specific font that renders Brahmi from ASCII input here */
  font-family: 'BrahmiFont', monospace; /* Placeholder font-family */
  font-size: 1.2rem;
  font-weight: bold;
}

.transliteration-display {
  font-family: 'Times New Roman', serif; /* Default for Latin */
  font-size: 1.1rem;
}

.devnagri-font {
  /* Assuming 'Tiro Devanagari Sanskrit' is linked globally and supports Devnagri ASCII */
  font-family: 'Tiro Devanagari Sanskrit', serif;
}

.transliteration-controls {
  display: flex;
  gap: 10px;
  margin-bottom: 15px;
}

.lang-button {
  padding: 8px 15px;
  border: 1px solid var(--color-border-clay);
  background-color: #e3d4bb;
  color: var(--color-text-dark);
  cursor: pointer;
  border-radius: 5px;
  transition: all 0.2s;
  font-weight: 500;
}

.lang-button:hover:not(.active) {
  background-color: #d1c1a9;
}

.lang-button.active {
  background-color: var(--color-highlight-gold);
  border-color: var(--color-highlight-gold);
  color: #fff;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
</style>