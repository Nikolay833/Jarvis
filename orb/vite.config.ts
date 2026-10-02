import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

// maplibre-gl 6 starts its web worker from `maplibre-gl-worker.mjs`, which imports `./maplibre-gl-shared.mjs`.
// A bundler would hash and separate them, so both are served/emitted untouched under /maplibre/.
const WORKER_FILES = ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"];
const dist = (f: string) => resolve(__dirname, "node_modules/maplibre-gl/dist", f);
function maplibreWorker(): Plugin {
  return {
    name: "maplibre-worker-files",
    configureServer(server) {
      server.middlewares.use("/maplibre", (req, res, next) => {
        const name = (req.url ?? "").replace(/^\//, "").split("?")[0];
        if (!WORKER_FILES.includes(name)) return next();
        res.setHeader("Content-Type", "text/javascript");
        res.end(readFileSync(dist(name)));
      });
    },
    generateBundle() {
      for (const f of WORKER_FILES) this.emitFile({ type: "asset", fileName: `maplibre/${f}`, source: readFileSync(dist(f)) });
    },
  };
}

// Tauri expects a fixed dev port and relative asset paths in the build.
// Two pages: the orb overlay (index.html) and the full-screen map window (map.html).
export default defineConfig({
  base: "./",
  clearScreen: false,
  plugins: [maplibreWorker()],
  server: { port: 1420, strictPort: true },
  build: {
    target: "es2021",
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200, // maplibre-gl is ~1 MB
    rollupOptions: {
      input: { main: resolve(__dirname, "index.html"), map: resolve(__dirname, "map.html") },
    },
  },
});
