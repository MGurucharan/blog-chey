import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
        secure: false,
      },
      "/openai-api":{
        target:"http://localhost:3001",
        changeOrigin:true,
        secure:false
      },
      "/fastapi":{
        target:"http://127.0.0.1:8000",
        changeOrigin:true,
        secure:false,
        rewrite: (path) => path.replace(/^\/fastapi/, ""),
      }
    },
  },
  base: process.env.VITE_BASE_PATH || "/blog-chey/"
});
