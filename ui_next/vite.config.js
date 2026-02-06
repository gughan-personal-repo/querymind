import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

const resolveFromUi = (pkgPath) => path.resolve(__dirname, "node_modules", pkgPath);

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      axios: resolveFromUi("axios/dist/esm/axios.js"),
      "lucide-react": resolveFromUi("lucide-react"),
      reactflow: resolveFromUi("reactflow"),
      "react-syntax-highlighter": resolveFromUi("react-syntax-highlighter"),
      react: resolveFromUi("react"),
      "react-dom": resolveFromUi("react-dom"),
    },
  },
  server: {
    fs: {
      allow: [".."],
    },
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
