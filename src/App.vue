<script setup>
import OCRPage from './views/OCRPage.vue';
</script>

<template>
  <div id="app-container">
    <OCRPage />
  </div>
</template>

<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@100;200;300;400;500;600;700;800;900&display=swap');

:root {
  --color-background-tan: #f5f5dc;
  --color-paper-sepia: #eaddc7;
  --color-border-clay: #b59f7d;
  --color-text-maroon: #800000;
  --color-highlight-gold: #cfa86e;
  --color-button-sepia: #9d7d5d;
  --color-button-hover: #7b5d43;
  --color-error: #a84c4c;
}

body {
  margin: 0;
  padding: 0;
  font-family: 'Poppins', sans-serif;
  color: var(--color-text-maroon);
}

#app-container {
  min-height: 100vh;
  position: relative;
  background-color: var(--color-background-tan);
  overflow: auto;
}

/* Fixed background with proper texture pattern */
#app-container::before {
  content: '';
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: var(--color-background-tan);
  background-image: 
    repeating-linear-gradient(
      45deg,
      transparent,
      transparent 10px,
      rgba(181, 159, 125, 0.03) 10px,
      rgba(181, 159, 125, 0.03) 20px
    ),
    repeating-linear-gradient(
      -45deg,
      transparent,
      transparent 10px,
      rgba(181, 159, 125, 0.03) 10px,
      rgba(181, 159, 125, 0.03) 20px
    );
  opacity: 0.8;
  z-index: -1;
}

h1, h2, h3 {
  font-weight: 400;
  border-bottom: 2px solid var(--color-border-clay);
  padding-bottom: 5px;
  margin-bottom: 15px;
}

button {
  background-color: var(--color-button-sepia);
  color: #fff;
  border: 1px solid var(--color-button-sepia);
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s, opacity 0.3s;
  box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
  font-family: inherit;
  font-size: 1rem;
}

button:hover:not(:disabled) {
  background-color: var(--color-button-hover);
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  box-shadow: none;
}

a {
  color: var(--color-highlight-gold);
}
</style>