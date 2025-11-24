<template>
  <div class="result-display glass-panel">
    <div class="header-row">
      <h2>Deciphered Results</h2>
      <button 
        v-if="results" 
        @click="downloadReport" 
        class="download-btn" 
        title="Download Report"
      >
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 16L12 8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M9 13L12 16L15 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M17 21H7C5.89543 21 5 20.1046 5 19V5C5 3.89543 5.89543 3 7 3H14L19 8V19C19 20.1046 18.1046 21 17 21Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Download PDF
      </button>
    </div>

    <div v-if="!results" class="placeholder">
      <div class="placeholder-content">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M14 2V8H20" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M16 13H8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M16 17H8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M10 9H8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <p>Upload an image to reveal its ancient secrets</p>
      </div>
    </div>

    <div v-else class="results-container">
      <div class="images-row">
        <div class="image-card">
          <h3>Original</h3>
          <div class="img-wrapper">
            <img :src="originalImageSrc" alt="Original Inscription" />
          </div>
        </div>
        <div class="image-card">
          <h3>Restored</h3>
          <div class="img-wrapper">
            <img :src="restoredImageSrc" alt="Restored Inscription" />
          </div>
        </div>
      </div>

      <div class="text-results">
        <div class="result-section">
          <h3>Recognized Brahmi</h3>
          <div class="recognized-text brahmi-script-display">
            <p>{{ results.recognized_brahmi_ascii }}</p>
          </div>
        </div>

        <div class="result-section">
          <div class="section-header">
            <h3>Transliteration</h3>
            <div class="toggle-controls">
              <button 
                :class="['toggle-btn', { active: isLatin }]" 
                @click="switchToLatin"
              >
                Latin
              </button>
              <button 
                :class="['toggle-btn', { active: !isLatin }]" 
                @click="switchToDevnagri"
                :disabled="isTransliterating"
              >
                {{ isTransliterating ? '...' : 'Devanagari' }}
              </button>
            </div>
          </div>
          
          <div class="recognized-text transliteration-display" :class="{ 'devnagri-font': !isLatin }">
            <p v-if="isLatin">{{ results.latin_transliteration }}</p>
            <p v-else>{{ devnagriTransliteration }}</p>
          </div>
        </div>
      </div>
      
      <div class="model-info">
        <small>Deciphered using: {{ results.model_used || 'Unknown Model' }}</small>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import transliterationService from '../services/transliterationService.js';
import { jsPDF } from "jspdf";

const props = defineProps({
  results: Object,
});

// Reactive state
const isLatin = ref(true);
const devnagriTransliteration = ref('');
const isTransliterating = ref(false);

// Computed properties for images
const restoredImageSrc = computed(() => {
  if (props.results && props.results.restored_image_b64) {
    return `data:image/jpeg;base64,${props.results.restored_image_b64}`;
  }
  return '';
});

const originalImageSrc = computed(() => {
  if (props.results && props.results.original_image_b64) {
    return `data:image/jpeg;base64,${props.results.original_image_b64}`;
  }
  return '';
});

// Switch to Latin
const switchToLatin = () => {
  isLatin.value = true;
};

// Switch to Devanagari with transliteration
const switchToDevnagri = async () => {
  isLatin.value = false;
  
  // If already transliterated, don't do it again
  if (devnagriTransliteration.value && props.results) {
    return;
  }

  // Transliterate the Latin text to Devanagari
  if (props.results && props.results.latin_transliteration) {
    isTransliterating.value = true;
    
    try {
      const latinText = props.results.latin_transliteration;
      const transliterated = await transliterationService.transliterate(latinText);
      devnagriTransliteration.value = transliterated;
    } catch (error) {
      console.error('Failed to transliterate:', error);
      devnagriTransliteration.value = `${props.results.latin_transliteration}\n\n[देवनागरी रूपांतरण विफल रहा]`;
    } finally {
      isTransliterating.value = false;
    }
  }
};

// Download Report Logic
const downloadReport = async () => {
  if (!props.results) return;

  const doc = new jsPDF();
  
  // Load Devanagari Font
  try {
    const fontUrl = 'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf';
    const response = await fetch(fontUrl);
    if (!response.ok) throw new Error('Failed to load font');
    const fontBuffer = await response.arrayBuffer();
    
    // Add font to VFS
    doc.addFileToVFS('NotoSansDevanagari-Regular.ttf', Buffer.from(fontBuffer).toString('binary'));
    doc.addFont('NotoSansDevanagari-Regular.ttf', 'NotoSansDevanagari', 'normal');
  } catch (e) {
    console.warn("Could not load Devanagari font, text may not render correctly:", e);
  }

  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 20;
  let yPos = 20;

  // Title
  doc.setFontSize(22);
  doc.setTextColor(128, 0, 0); // Maroon
  doc.text("Brahmi OCR Report", pageWidth / 2, yPos, { align: "center" });
  yPos += 15;

  doc.setFontSize(12);
  doc.setTextColor(0, 0, 0);
  doc.text(`Date: ${new Date().toLocaleDateString()}`, margin, yPos);
  yPos += 10;
  doc.text(`Model Used: ${props.results.model_used || 'Unknown'}`, margin, yPos);
  yPos += 15;

  // Images
  const imgWidth = 70;
  const imgHeight = 40;
  
  if (props.results.original_image_b64) {
    doc.text("Original Inscription:", margin, yPos);
    doc.addImage(originalImageSrc.value, "JPEG", margin, yPos + 5, imgWidth, imgHeight);
  }
  
  if (props.results.restored_image_b64) {
    doc.text("Restored Inscription:", pageWidth / 2 + 10, yPos);
    doc.addImage(restoredImageSrc.value, "JPEG", pageWidth / 2 + 10, yPos + 5, imgWidth, imgHeight);
  }
  
  yPos += imgHeight + 20;

  // Results
  doc.setLineWidth(0.5);
  doc.line(margin, yPos, pageWidth - margin, yPos);
  yPos += 10;

  doc.setFontSize(16);
  doc.setTextColor(128, 0, 0);
  doc.text("Deciphered Text", margin, yPos);
  yPos += 10;

  doc.setFontSize(12);
  doc.setTextColor(0, 0, 0);
  
  // Brahmi
  doc.setFont("helvetica", "bold");
  doc.text("Recognized Brahmi (ASCII/Info):", margin, yPos);
  yPos += 7;
  doc.setFont("helvetica", "normal");
  const brahmiLines = doc.splitTextToSize(props.results.recognized_brahmi_ascii, pageWidth - 2 * margin);
  doc.text(brahmiLines, margin, yPos);
  yPos += (brahmiLines.length * 7) + 10;

  // Latin
  doc.setFont("helvetica", "bold");
  doc.text("Latin Transliteration:", margin, yPos);
  yPos += 7;
  doc.setFont("helvetica", "normal");
  const latinLines = doc.splitTextToSize(props.results.latin_transliteration, pageWidth - 2 * margin);
  doc.text(latinLines, margin, yPos);
  yPos += (latinLines.length * 7) + 10;

  // Devanagari
  if (devnagriTransliteration.value) {
      doc.setFont("helvetica", "bold");
      doc.text("Devanagari Transliteration:", margin, yPos);
      yPos += 7;
      
      // Switch to Devanagari font
      doc.setFont("NotoSansDevanagari", "normal");
      const devLines = doc.splitTextToSize(devnagriTransliteration.value, pageWidth - 2 * margin);
      doc.text(devLines, margin, yPos);
      
      // Revert to standard font
      doc.setFont("helvetica", "normal");
  }

  // Save
  doc.save("brahmi_ocr_report.pdf");
};

// Watch for results changes and reset
watch(() => props.results, (newResults) => {
  if (newResults) {
    isLatin.value = true;
    devnagriTransliteration.value = '';
  }
});
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
  overflow-y: auto;
}

.header-row {
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
  margin: 0;
  font-weight: 700;
}

.download-btn {
  background: rgba(255, 255, 255, 0.5);
  border: 1px solid var(--color-gold-accent);
  color: var(--color-royal-maroon);
  padding: 6px 12px;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.9rem;
  font-weight: 600;
  transition: all 0.2s;
}

.download-btn:hover {
  background: var(--color-royal-maroon);
  color: var(--color-antique-cream);
}

.download-btn svg {
  width: 16px;
  height: 16px;
}

h3 {
  font-size: 1rem;
  color: var(--color-text-secondary);
  margin-bottom: 8px;
  font-family: 'Outfit', sans-serif;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 600;
}

.placeholder {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-secondary);
  border: 2px dashed var(--color-glass-border);
  border-radius: 12px;
  min-height: 300px;
  background: rgba(255, 255, 255, 0.1);
}

.placeholder-content {
  text-align: center;
  opacity: 0.7;
}

.placeholder svg {
  width: 64px;
  height: 64px;
  margin-bottom: 16px;
  color: var(--color-gold-accent);
}

.results-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
  flex: 1;
}

.images-row {
  display: flex;
  gap: 16px;
}

.image-card {
  flex: 1;
}

.img-wrapper {
  background: rgba(255, 255, 255, 0.4);
  border-radius: 8px;
  padding: 8px;
  border: 1px solid var(--color-glass-border);
  height: 150px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: inset 0 2px 6px rgba(0,0,0,0.05);
}

.img-wrapper img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.text-results {
  display: flex;
  flex-direction: column;
  gap: 16px;
  flex: 1;
}

.result-section {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 12px;
  padding: 16px;
  border: 1px solid var(--color-glass-border);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.section-header h3 {
  margin: 0;
}

.toggle-controls {
  display: flex;
  background: rgba(255, 255, 255, 0.4);
  border-radius: 20px;
  padding: 2px;
  border: 1px solid var(--color-glass-border);
}

.toggle-btn {
  background: transparent;
  border: none;
  color: var(--color-text-secondary);
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 0.8rem;
  font-weight: 500;
  cursor: pointer;
}

.toggle-btn.active {
  background: var(--color-royal-maroon);
  color: var(--color-antique-cream);
  font-weight: 600;
}

.recognized-text {
  color: var(--color-royal-maroon);
  font-size: 1.1rem;
  white-space: pre-wrap;
  line-height: 1.6;
}

.brahmi-script-display {
  font-family: 'BrahmiFont', monospace;
  color: var(--color-deep-saffron);
  font-weight: 600;
}

.transliteration-display {
  font-family: 'Outfit', sans-serif;
}

.devnagri-font {
  font-family: 'Tiro Devanagari Sanskrit', serif;
  font-size: 1.3rem;
}

.model-info {
  text-align: right;
  color: var(--color-text-secondary);
  font-size: 0.8rem;
  margin-top: auto;
  opacity: 0.7;
}

/* Scrollbar for the panel content */
.glass-panel::-webkit-scrollbar {
  width: 6px;
}

.glass-panel::-webkit-scrollbar-track {
  background: transparent;
}

.glass-panel::-webkit-scrollbar-thumb {
  background: rgba(74, 4, 4, 0.1);
  border-radius: 3px;
}

.glass-panel::-webkit-scrollbar-thumb:hover {
  background: rgba(74, 4, 4, 0.2);
}
</style>