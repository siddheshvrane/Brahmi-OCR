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
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

:root {
  /* Kaavi Art Minimalist Palette */
  --color-kaavi-plaster: #FDFBF7; /* Clean, warm white plaster */
  --color-kaavi-red: #8B2C24;     /* Dominant Kaavi Terracotta */
  --color-kaavi-earth: #B34E42;   /* Lighter earth variation */
  
  /* Neutral scale */
  --color-surface: #FFFFFF;
  --color-surface-hover: #FFF9F8;
  
  /* Text */
  --color-text-primary: #381B18;  /* Very dark warm brown/red */
  --color-text-secondary: #7A534E; /* Muted earth tone for secondary text */
  --color-border: #E8D8D5;      /* Plaster tinted border */
  
  /* Bounding Box Action Color */
  --color-success: #10b981; /* Precise Emerald green for boxes */
  
  /* Elevation */
  --shadow-sm: 0 1px 2px 0 rgba(139, 44, 36, 0.05); /* Tinted subtle shadow */
  --shadow-md: 0 4px 6px -1px rgba(139, 44, 36, 0.05), 0 2px 4px -2px rgba(139, 44, 36, 0.05);
  --shadow-lg: 0 10px 25px -3px rgba(139, 44, 36, 0.08), 0 4px 6px -4px rgba(139, 44, 36, 0.04);
  --radius-md: 8px; /* Tighter borders for minimalist structural look */
  --radius-lg: 12px;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
  font-family: 'Poppins', system-ui, -apple-system, sans-serif;
  color: var(--color-text-primary);
  background-color: var(--color-kaavi-plaster);
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
}

#app-container {
  width: 100vw;
  height: 100vh;
  position: relative;
  display: flex;
  justify-content: center;
}

/* Subtle Kaavi Geometric Accent (CSS pattern) */
.kaavi-border-accent {
  position: relative;
}
.kaavi-border-accent::after {
  content: "";
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 6px;
  background-image: linear-gradient(135deg, var(--color-kaavi-red) 25%, transparent 25%), 
                    linear-gradient(225deg, var(--color-kaavi-red) 25%, transparent 25%), 
                    linear-gradient(45deg, var(--color-kaavi-red) 25%, transparent 25%), 
                    linear-gradient(315deg, var(--color-kaavi-red) 25%, var(--color-surface) 25%);
  background-position: 10px 0, 10px 0, 0 0, 0 0;
  background-size: 20px 20px;
  background-repeat: repeat-x;
  border-bottom-left-radius: var(--radius-lg);
  border-bottom-right-radius: var(--radius-lg);
  opacity: 0.15; /* Extremely subtle */
}

/* Global Scrollbar Formatting */
::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: var(--color-border);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--color-kaavi-earth);
}

button {
  font-family: 'Poppins', system-ui, -apple-system, sans-serif;
  cursor: pointer;
  transition: all 0.2s ease;
}
</style>