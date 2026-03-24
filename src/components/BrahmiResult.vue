<template>
  <div class="result-card">
    <!-- Header Section -->
    <div class="card-header">
      <div class="header-text">
        <h2>Deciphered Results</h2>
        <p class="subtitle">Interactive character refinement & identification</p>
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="!initialData" class="empty-state">
      <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <rect x="3" y="3" width="18" height="18" rx="2" stroke-width="1.5"/>
        <path d="M3 14l4-4 11 11" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="8" cy="8" r="2" stroke-width="1.5"/>
      </svg>
      <p>Results will appear here once processing begins</p>
    </div>

    <!-- Active Results -->
    <div v-else class="content-wrapper">
      <div class="canvas-card kaavi-border-accent">
        <div class="toolbar">
          <div class="toolbar-left">
            <h3>{{ phase === 'segmentation' ? 'Detected Boxes' : 'GAN-Restored Image' }}</h3>
            <span class="badge" :class="phase === 'segmentation' ? 'badge-review' : ''">
              {{ phase === 'segmentation' ? 'Review Boxes' : 'Refine' }}
            </span>
          </div>
          <div class="toolbar-right">
            <label class="switch-container">
              <span class="switch-label">Boxes</span>
              <div class="toggle-switch">
                <input type="checkbox" v-model="showBoundingBoxes" class="sr-only">
                <div class="toggle-bg" :class="{ 'active': showBoundingBoxes }"></div>
                <div class="toggle-knob" :class="{ 'active': showBoundingBoxes }"></div>
              </div>
            </label>
            <div class="divider"></div>
            <button @click="clearAllBoxes" class="btn-icon" title="Clear all boxes">
              <svg viewBox="0 0 24 24" fill="none" width="16" height="16" stroke="currentColor">
                <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" stroke-width="2"/>
              </svg>
            </button>
          </div>
        </div>
        
        <div class="canvas-container" ref="canvasWrapper">
          <canvas 
            ref="interactiveCanvas" 
            @mousedown="startDrawing" 
            @mousemove="draw" 
            @mouseup="stopDrawing"
            @mouseleave="cancelDrawing"
            :class="{ 'is-drawing': isDrawing, 'pointer-events-none': !showBoundingBoxes }"
            class="interactive-canvas"
          ></canvas>
          <div v-if="!showBoundingBoxes" class="overlay-hint">
            Enable boxes to refine selection
          </div>
        </div>
        <div class="canvas-footer">
          Click a box to delete • Drag to draw new box
        </div>
      </div>

      <!-- Phase 1: Segmentation — Apply GAN button -->
      <div v-if="phase === 'segmentation'" class="action-bar kaavi-border-accent">
        <div class="phase-hint">
          <svg viewBox="0 0 24 24" fill="none" width="14" height="14" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <span>Review boxes, edit if needed, then apply GAN restoration</span>
        </div>
        <button
          @click="applyGAN"
          :disabled="isApplyingGAN || currentBoxes.length === 0"
          class="btn-primary"
        >
          <svg v-if="isApplyingGAN" class="spinner" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-dasharray="32" stroke-dashoffset="0" stroke-linecap="round"></circle>
          </svg>
          <span>{{ isApplyingGAN ? 'Restoring...' : 'Apply GAN Restoration' }}</span>
        </button>
      </div>

      <!-- Phase 2: Restoration — Decipher button -->
      <div v-if="phase === 'restoration'" class="action-bar kaavi-border-accent">
        <div class="input-group">
          <label>OCR Engine</label>
          <div class="select-wrapper">
            <select v-model="selectedModel" class="form-select">
              <option value="ResNet50">ResNet50</option>
              <option value="EfficientNetB0">EfficientNetB0</option>
              <option value="MobileNetV2">MobileNetV2</option>
              <option value="Ensemble">Ensemble AI (Recommended)</option>
            </select>
            <svg class="select-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M6 9l6 6 6-6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
        </div>
        <button 
          @click="decipherCharacters" 
          :disabled="isDeciphering || currentBoxes.length === 0" 
          class="btn-primary"
        >
          <svg v-if="isDeciphering" class="spinner" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-dasharray="32" stroke-dashoffset="0" stroke-linecap="round"></circle>
          </svg>
          <span>{{ isDeciphering ? 'Analyzing...' : 'Decipher Characters' }}</span>
        </button>
      </div>

      <!-- Result Sections -->
      <div v-if="predictionResults" class="results-stack">
        <!-- Transliteration Card -->
        <div class="result-tile kaavi-border-accent">
          <div class="tile-header header-spread">
            <h4>Transliteration & Script</h4>
            <div class="segmented-control">
              <button :class="{ active: transliterationMode === 'latin' }" @click="transliterationMode = 'latin'">Latin</button>
              <button :class="{ active: transliterationMode === 'devanagari' }" @click="transliterationMode = 'devanagari'">Devanagari</button>
              <button :class="{ active: transliterationMode === 'brahmi' }" @click="transliterationMode = 'brahmi'">Brahmi</button>
            </div>
          </div>
          <div class="tile-content" :class="{ 'devanagari-font': transliterationMode === 'devanagari', 'brahmi-font': transliterationMode === 'brahmi' }">
            {{ 
              transliterationMode === 'latin' ? predictionResults.top_prediction : 
              transliterationMode === 'devanagari' ? predictionResults.top_prediction_devanagari :
              predictionResults.top_prediction_brahmi
            }}
          </div>
        </div>
        
        <div class="meta-footer">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>Processed via {{ predictionResults.model_used }} • Average Confidence: {{ predictionResults.top_confidence.toFixed(1) }}%</span>
        </div>

        <div class="report-action-wrapper">
          <button 
            @click="downloadReport" 
            class="btn-outline-primary" 
          >
            <svg viewBox="0 0 24 24" fill="none" width="16" height="16" stroke="currentColor">
              <path d="M12 16L12 8M9 13L12 16L15 13M17 21H7C5.89543 21 5 20.1046 5 19V5C5 3.89543 5.89543 3 7 3H14L19 8V19C19 20.1046 18.1046 21 17 21Z" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>Download PDF Report</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch, onMounted, onUnmounted } from 'vue';
import { jsPDF } from "jspdf";

const props = defineProps({
  initialData: Object,
});

// --- State ---
const API_URL = 'http://localhost:5000';
const transliterationMode = ref('latin'); // 'latin', 'devanagari', 'brahmi'
const isDeciphering = ref(false);
const isApplyingGAN = ref(false);
const showBoundingBoxes = ref(true);
const selectedModel = ref('Ensemble');
const currentBoxes = ref([]); 
const predictionResults = ref(null);
const phase = ref('segmentation'); // 'segmentation' | 'restoration'

// Canvas Refs & State
const interactiveCanvas = ref(null);
const canvasWrapper = ref(null);
const isDrawing = ref(false);
const startX = ref(0);
const startY = ref(0);
const currentX = ref(0);
const currentY = ref(0);

// Source images — canvas shows different image depending on phase
const canvasImageSrc = computed(() => {
  if (!props.initialData) return '';
  // In segmentation phase: show original (no GAN). In restoration: show GAN result.
  const b64 = phase.value === 'restoration' && props.initialData.restored_image_b64
    ? props.initialData.restored_image_b64
    : props.initialData.original_image_b64;
  return `data:image/jpeg;base64,${b64}`;
});

// Image object for canvas operations
let offscreenImg = null;

// --- Canvas Logic ---

const initCanvas = () => {
  if (!props.initialData || !interactiveCanvas.value || !canvasWrapper.value) return;
  
  // Always reset boxes from the new image data so old boxes don't bleed into new images
  currentBoxes.value = [...props.initialData.initial_boxes];

  const img = new Image();
  img.onload = () => {
    offscreenImg = img;
    const canvas = interactiveCanvas.value;
    
    // Fit canvas to wrapper while maintaining aspect ratio
    const wrapper = canvasWrapper.value;
    const wrapperWidth = wrapper.clientWidth;
    const wrapperHeight = 400; // Fixed max height for the drawing area
    
    const scale = Math.min(wrapperWidth / img.width, wrapperHeight / img.height);
    
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    
    renderCanvas();
  };
  img.src = canvasImageSrc.value;
};

const renderCanvas = () => {
  if (!interactiveCanvas.value || !offscreenImg) return;
  
  const ctx = interactiveCanvas.value.getContext('2d');
  const canvas = interactiveCanvas.value;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(offscreenImg, 0, 0, canvas.width, canvas.height);
  
  if (!showBoundingBoxes.value) return;

  const scale = canvas.width / offscreenImg.width;

  // Draw Boxes
  ctx.lineWidth = 2;
  currentBoxes.value.forEach((box) => {
    const [x, y, w, h] = box;
    ctx.strokeStyle = '#10b981'; // Emerald Green
    ctx.strokeRect(x * scale, y * scale, w * scale, h * scale);
  });
  
  if (isDrawing.value) {
    ctx.strokeStyle = '#8B2C24'; // Kaavi Terracotta Red for drawing
    ctx.setLineDash([4, 4]);
    const rectX = Math.min(startX.value, currentX.value);
    const rectY = Math.min(startY.value, currentY.value);
    const rectW = Math.abs(currentX.value - startX.value);
    const rectH = Math.abs(currentY.value - startY.value);
    ctx.strokeRect(rectX, rectY, rectW, rectH);
    ctx.setLineDash([]);
  }
};

const startDrawing = (e) => {
  if (!showBoundingBoxes.value) return;
  const rect = interactiveCanvas.value.getBoundingClientRect();
  isDrawing.value = true;
  startX.value = e.clientX - rect.left;
  startY.value = e.clientY - rect.top;
  currentX.value = startX.value;
  currentY.value = startY.value;
};

const draw = (e) => {
  if (!isDrawing.value) return;
  const rect = interactiveCanvas.value.getBoundingClientRect();
  currentX.value = e.clientX - rect.left;
  currentY.value = e.clientY - rect.top;
  renderCanvas();
};

const stopDrawing = (e) => {
  if (!isDrawing.value) return;
  isDrawing.value = false;
  
  const rect = interactiveCanvas.value.getBoundingClientRect();
  const endX = e.clientX - rect.left;
  const endY = e.clientY - rect.top;
  
  const w = Math.abs(endX - startX.value);
  const h = Math.abs(endY - startY.value);
  
  if (w > 5 && h > 5) {
    const scale = offscreenImg.width / interactiveCanvas.value.width;
    let realX = Math.min(startX.value, endX) * scale;
    let realY = Math.min(startY.value, endY) * scale;
    let realW = w * scale;
    let realH = h * scale;

    // Normalise to dataset aspect ratio (width / height = 0.7145)
    const TARGET_AR = 0.7145;
    const imgW = offscreenImg.width;
    const imgH = offscreenImg.height;
    const currentAR = realW / realH;
    if (currentAR < TARGET_AR) {
      // Too tall — expand width
      const newW = realH * TARGET_AR;
      realX = realX - (newW - realW) / 2;
      realW = newW;
    } else {
      // Too wide — expand height
      const newH = realW / TARGET_AR;
      realY = realY - (newH - realH) / 2;
      realH = newH;
    }
    // Clamp to image bounds
    realX = Math.max(0, realX);
    realY = Math.max(0, realY);
    realW = Math.min(imgW - realX, realW);
    realH = Math.min(imgH - realY, realH);

    const newBox = [Math.round(realX), Math.round(realY), Math.round(realW), Math.round(realH)];

    // Insert the new box at the correct reading-order position (left-to-right by X)
    // so that when sent to the backend the character order matches the visual layout.
    const insertIdx = currentBoxes.value.findIndex(([bx]) => bx > newBox[0]);
    if (insertIdx === -1) {
      currentBoxes.value.push(newBox);
    } else {
      currentBoxes.value.splice(insertIdx, 0, newBox);
    }

    renderCanvas();
  } else {
    // Treat as a click (delete box)
    const scale = offscreenImg.width / interactiveCanvas.value.width;
    const realX = endX * scale;
    const realY = endY * scale;

    let boxToRemove = -1;
    for (let i = currentBoxes.value.length - 1; i >= 0; i--) {
      const [x, y, w, h] = currentBoxes.value[i];
      if (realX >= x && realX <= x + w && realY >= y && realY <= y + h) {
        boxToRemove = i;
        break;
      }
    }

    if (boxToRemove !== -1) {
      currentBoxes.value.splice(boxToRemove, 1);
      renderCanvas();
    }
  }
};

const cancelDrawing = () => {
  if (isDrawing.value) {
    isDrawing.value = false;
    renderCanvas();
  }
};

const clearAllBoxes = () => {
  currentBoxes.value = [];
  renderCanvas();
};

// --- API ---

// Phase 1 → Phase 2: call /process with user-confirmed boxes
const applyGAN = async () => {
  if (currentBoxes.value.length === 0) return;
  isApplyingGAN.value = true;

  try {
    const response = await fetch(`${API_URL}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: props.initialData.original_image_b64,
        boxes: currentBoxes.value
      })
    });
    const data = await response.json();
    if (data.success) {
      // 1) Change local phase first so canvasImageSrc switches
      phase.value = 'restoration';
      
      // 2) Mutate initialData silently (watcher ignores this because phase != 'segmentation')
      props.initialData.phase = 'restoration';
      props.initialData.restored_image_b64 = data.restored_image_b64;
      props.initialData.initial_boxes = data.boxes;
      currentBoxes.value = [...data.boxes];
      
      // 3) Reload canvas with GAN-restored image
      const img = new Image();
      img.onload = () => { offscreenImg = img; renderCanvas(); };
      img.src = `data:image/jpeg;base64,${data.restored_image_b64}`;
    } else {
      console.error('GAN restoration failed:', data.error);
    }
  } catch (err) {
    console.error('applyGAN error:', err);
  } finally {
    isApplyingGAN.value = false;
  }
};

const decipherCharacters = async () => {
  if (currentBoxes.value.length === 0) return;
  isDeciphering.value = true;
  predictionResults.value = null;

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        image: props.initialData.original_image_b64,
        model: selectedModel.value,
        boxes: currentBoxes.value
      })
    });

    const data = await response.json();
    if (data.success) {
      predictionResults.value = data;
    } else {
      console.error(data.error);
    }
  } catch (err) {
    console.error(err);
  } finally {
    isDeciphering.value = false;
  }
};

const downloadReport = async () => {
  if (!predictionResults.value) return;
  const doc = new jsPDF();
  doc.setFontSize(22);
  doc.setTextColor(28, 25, 23);
  doc.text("Brahmi OCR Analysis", 105, 20, { align: "center" });
  doc.save("brahmi_ocr_report.pdf");
};

// --- Lifecycle ---

watch(() => props.initialData, (newData, oldData) => {
  // Only completely reset if it's explicitly a NEW upload (from FileUploader)
  // FileUploader now explicitly sets phase: 'segmentation'
  if (newData && newData.phase === 'segmentation') {
    predictionResults.value = null;
    phase.value = 'segmentation';
    setTimeout(initCanvas, 100);
  }
}, { deep: true });

watch(showBoundingBoxes, () => renderCanvas());

onMounted(() => {
  if (props.initialData) initCanvas();
  window.addEventListener('resize', initCanvas);
});

onUnmounted(() => window.removeEventListener('resize', initCanvas));
</script>

<style scoped>
.result-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

/* Header */
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.header-text h2 {
  font-size: 1.25rem;
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

.btn-icon-text {
  display: flex;
  align-items: center;
  gap: 6px;
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 8px 12px;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--color-text-primary);
  box-shadow: var(--shadow-sm);
}

.btn-icon-text:hover {
  background-color: var(--color-surface-hover);
}

/* Empty State */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: var(--color-surface);
  border: 1px dashed var(--color-border);
  border-radius: var(--radius-lg);
  padding: 40px;
}

.empty-icon {
  width: 48px;
  height: 48px;
  color: var(--color-text-secondary);
  margin-bottom: 16px;
  opacity: 0.5;
}

.empty-state p {
  color: var(--color-text-secondary);
  font-size: 0.875rem;
  margin: 0;
}

/* Content Wrapper */
.content-wrapper {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* Canvas Card */
.canvas-card {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.toolbar {
  padding: 12px 16px;
  border-bottom: 1px solid var(--color-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #fafaf9; /* Stone 50 */
}

.toolbar-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.toolbar-left h3 {
  font-size: 0.875rem;
  font-weight: 600;
  margin: 0;
  color: var(--color-text-primary);
}

.badge {
  background-color: rgba(139, 44, 36, 0.05); /* Kaavi red tint */
  color: var(--color-kaavi-red);
  font-size: 0.75rem;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 9999px;
  letter-spacing: 0.5px;
}

.badge-review {
  background-color: rgba(59, 130, 246, 0.1); /* Blue tint */
  color: #2563eb; /* Blue 600 */
}

.toolbar-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* Switch */
.switch-container {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.switch-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--color-text-secondary);
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}

.toggle-switch {
  position: relative;
  width: 36px;
  height: 20px;
}

.toggle-bg {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #d6d3d1; /* Stone 300 */
  border-radius: 9999px;
  transition: background-color 0.2s ease;
}

.toggle-bg.active {
  background-color: var(--color-kaavi-red);
}

.toggle-knob {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 16px;
  height: 16px;
  background-color: white;
  border-radius: 50%;
  transition: transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.toggle-knob.active {
  transform: translateX(16px);
}

.divider {
  width: 1px;
  height: 16px;
  background-color: var(--color-border);
}

.btn-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  border: none;
  background: transparent;
  color: var(--color-text-secondary);
  transition: all 0.2s ease;
}

.btn-icon:hover {
  background-color: #fee2e2;
  color: #b91c1c;
}

.canvas-container {
  width: 100%;
  height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  background-color: #f5f5f4; /* Stone 100 */
  border-bottom: 1px solid var(--color-border);
}

.interactive-canvas {
  max-width: 100%;
  max-height: 100%;
}

.is-drawing {
  cursor: crosshair;
}

.pointer-events-none {
  pointer-events: none;
}

.overlay-hint {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background-color: rgba(28, 25, 23, 0.7);
  color: white;
  padding: 8px 16px;
  border-radius: 9999px;
  font-size: 0.875rem;
  font-weight: 500;
  backdrop-filter: blur(4px);
  pointer-events: none;
}

.canvas-footer {
  padding: 8px 16px;
  background-color: white;
  font-size: 0.75rem;
  color: var(--color-text-secondary);
  text-align: center;
}

/* Action Bar */
.action-bar {
  display: flex;
  align-items: flex-end;
  justify-content: flex-end;
  gap: 16px;
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 16px;
  box-shadow: var(--shadow-sm);
}

.phase-hint {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--color-text-secondary);
  font-size: 0.875rem;
  margin-right: auto;
  align-self: center;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex: 1;
}

.input-group label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.select-wrapper {
  position: relative;
}

.form-select {
  width: 100%;
  appearance: none;
  background-color: var(--color-bg-app);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 10px 36px 10px 12px;
  font-family: inherit;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--color-text-primary);
  outline: none;
  cursor: pointer;
  transition: border-color 0.2s;
}

.form-select:hover, .form-select:focus {
  border-color: #a8a29e;
}

.select-icon {
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  width: 16px;
  height: 16px;
  color: var(--color-text-secondary);
  pointer-events: none;
}

.btn-primary {
  background-color: var(--color-kaavi-red);
  color: white;
  border: none;
  border-radius: var(--radius-md);
  padding: 10px 24px;
  font-size: 0.95rem;
  font-weight: 500;
  letter-spacing: 0.5px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  box-shadow: var(--shadow-sm);
  white-space: nowrap;
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

/* Results Stack */
.results-stack {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.result-tile {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 20px;
  box-shadow: var(--shadow-sm);
}

.tile-header {
  margin-bottom: 12px;
}

.header-spread {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tile-header h4 {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin: 0;
}

.tile-content {
  font-size: 1.5rem;
  font-weight: 500;
  color: var(--color-text-primary);
  word-break: break-all;
}

.brahmi-font {
  font-family: "Noto Sans Brahmi", "Segoe UI Historic", sans-serif;
  font-size: 2rem;
  letter-spacing: 4px;
  line-height: 1.4;
  color: var(--color-text-primary);
}

.devanagari-font {
  font-family: inherit;
}

.segmented-control {
  display: flex;
  background-color: var(--color-bg-app);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 2px;
}

.segmented-control button {
  background: transparent;
  border: none;
  padding: 4px 12px;
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--color-text-secondary);
  border-radius: 4px;
}

.segmented-control button.active {
  background-color: white;
  color: var(--color-text-primary);
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.meta-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-size: 0.75rem;
  color: var(--color-text-secondary);
  margin-top: 8px;
}

.report-action-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 8px;
  padding-top: 16px;
  border-top: 1px dashed var(--color-border);
}

.btn-outline-primary {
  display: flex;
  align-items: center;
  gap: 8px;
  background-color: transparent;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 12px 28px;
  font-size: 0.95rem;
  font-weight: 500;
  letter-spacing: 0.5px;
  color: var(--color-kaavi-red);
  transition: all 0.2s ease;
}

.btn-outline-primary:hover {
  background-color: rgba(139, 44, 36, 0.03); /* Kaavi tint */
  border-color: var(--color-kaavi-earth);
}

@media (max-width: 640px) {
  .action-bar {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
