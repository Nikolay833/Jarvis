import { defineConfig } from "vite";

// Tauri expects a fixed dev port and relative asset paths in the build.
export default defineConfig({
  base: "./",
  clearScreen: false,
  server: { port: 1420, strictPort: true },
  build: { target: "es2021", outDir: "dist", emptyOutDir: true },
});
