/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        aura: {
          bg: '#05070d',
          panel: 'rgba(16,25,46,0.6)',
          accent: '#19b7ff',
          accent2: '#6d3bff',
          danger: '#ff3b5c',
          warn: '#ffc53b',
          success: '#35d07f'
        }
      },
      boxShadow: {
        'aura-glow': '0 0 12px 2px rgba(25,183,255,0.4)',
        'aura-inner': 'inset 0 0 12px rgba(109,59,255,0.25)'
      },
      backdropBlur: {
        xs: '2px'
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        display: ['"Orbitron"', 'sans-serif']
      }
    }
  },
  plugins: []
};
