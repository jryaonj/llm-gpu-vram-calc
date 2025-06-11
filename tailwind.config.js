/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{ts,tsx,js,jsx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [require('daisyui')],
  // optional: pick one of Daisy’s themes, or leave default
  daisyui: {
    themes: ['light', 'dark', 'cupcake'],
  },
}
