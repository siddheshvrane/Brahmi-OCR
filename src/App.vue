<template>
  <div id="app-container">
    <div class="glass-overlay"></div>
    <OCRPage />
  </div>
</template>

<script setup>
import OCRPage from './views/OCRPage.vue';
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Outfit:wght@300;400;500;600&display=swap');

:root {
  /* Ancient Indian Palette - Earthy & Creamy */
  --color-deep-saffron: #E2725B; /* Terracotta/Saffron */
  --color-royal-maroon: #4A0404; /* Dark Maroon Text */
  --color-gold-accent: #D4AF37; /* Antique Gold */
  --color-antique-cream: #FAF0E6; /* Linen/Cream Background */
  --color-parchment: #F5E6D3; /* Darker Cream for panels */
  
  /* Glassmorphism - Warm Tint */
  --color-glass-bg: rgba(255, 253, 240, 0.6); /* Milky/Creamy Glass */
  --color-glass-border: rgba(212, 175, 55, 0.3); /* Gold Border */
  
  --color-text-primary: #4A0404; /* Maroon */
  --color-text-secondary: #8B4513; /* Saddle Brown */
  
  /* Shadows */
  --shadow-glass: 0 8px 32px 0 rgba(74, 4, 4, 0.15);
}

body {
  margin: 0;
  padding: 0;
  font-family: 'Outfit', sans-serif;
  color: var(--color-text-primary);
  background-color: var(--color-antique-cream);
  overflow: hidden;
}

#app-container {
  width: 100vw;
  height: 100vh;
  position: relative;
  /* Earthy texture or gradient background */
  background: radial-gradient(circle at center, #FAF0E6 0%, #F5E6D3 100%);
  overflow: hidden;
}

.glass-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: url('https://www.transparenttextures.com/patterns/aged-paper.png'); /* Subtle paper texture */
  opacity: 0.4;
  pointer-events: none;
  z-index: 0;
}

/* Global Scrollbar Styling for internal containers */
::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
}

::-webkit-scrollbar-thumb {
  background: var(--color-deep-saffron);
  border-radius: 3px;
}

h1, h2, h3 {
  font-family: 'Cinzel Decorative', cursive;
  color: var(--color-royal-maroon);
  margin: 0;
}

button {
  font-family: 'Outfit', sans-serif;
  cursor: pointer;
  transition: all 0.3s ease;
}
</style>