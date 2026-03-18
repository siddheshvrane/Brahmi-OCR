<template>
  <div class="ocr-page">
    <header class="app-header">
      <div class="brand">
        <img :src="logoImg" alt="Brahmi OCR Logo" class="brand-logo" />
        <div class="brand-text">
          <h1>Brahmi OCR</h1>
          <p>Rediscovering Ancient Wisdom through AI</p>
        </div>
      </div>
    </header>
    
    <main class="app-main">
      <aside class="sidebar">
        <FileUploader @image-processed="handleImageProcessed" />
      </aside>

      <section class="content-area">
        <BrahmiResult :initialData="processedData" />
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import FileUploader from '../components/FileUploader.vue';
import BrahmiResult from '../components/BrahmiResult.vue';
import logoImg from '../assets/logo.png';

const processedData = ref(null);

const handleImageProcessed = (data) => {
  processedData.value = data;
};
</script>

<style scoped>
.ocr-page {
  width: 100%;
  height: 100vh;
  max-width: 1440px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  padding: 32px 48px;
  gap: 32px;
}

/* Header */
.app-header {
  flex-shrink: 0;
}

.brand {
  display: flex;
  align-items: center;
  gap: 20px;
}

.brand-logo {
  height: 56px;
  width: auto;
  opacity: 0.9;
}

.brand-text h1 {
  font-family: inherit;
  font-size: 1.75rem;
  color: var(--color-kaavi-red);
  margin: 0;
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -0.02em;
}

.brand-text p {
  color: var(--color-text-secondary);
  font-size: 0.875rem;
  margin: 4px 0 0 0;
  font-weight: 500;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

/* Layout */
.app-main {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
  flex: 1;
  min-height: 0;
}

.sidebar {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.content-area {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

/* Responsive Design */
@media (max-width: 1024px) {
  .ocr-page {
    height: auto;
    padding: 24px;
    gap: 24px;
  }

  .app-main {
    grid-template-columns: 1fr;
    height: auto;
  }

  .sidebar {
    width: 100%;
    height: auto;
  }

  .content-area {
    min-height: 600px;
  }
}
</style>