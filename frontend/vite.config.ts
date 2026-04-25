import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/ws': {
        target: 'http://localhost:8089',
        ws: true,
      },
      '/chat': 'http://localhost:8089',
      '/documents': 'http://localhost:8089',
      '/health': 'http://localhost:8089',
    },
  },
})
