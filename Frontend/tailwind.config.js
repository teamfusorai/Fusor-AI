/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Custom dark theme colors matching your design
        'zinc': {
          50: '#fafafa',
          100: '#f4f4f5',
          200: '#e4e4e7',
          300: '#d4d4d8',
          400: '#a1a1aa',
          500: '#71717a',
          600: '#52525b',
          700: '#3f3f46',
          800: '#27272a',
          900: '#18181b',
          950: '#09090b',
        },
        'indigo': {
          400: '#818cf8',
          500: '#6366f1',
          600: '#4f46e5',
          700: '#4338ca',
        },
        'amber': {
          500: '#f59e0b',
        },
        'emerald': {
          500: '#10b981',
        },
        'blue': {
          500: '#3b82f6',
        },
        'red': {
          400: '#f87171',
          500: '#ef4444',
        },
        'green': {
          400: '#4ade80',
          500: '#22c55e',
        },
      },
    },
  },
  plugins: [],
}
