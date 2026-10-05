import { resolve } from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // 监听所有网卡：外网端口映射（如 3000）需要能连进来；
    // 默认只绑 ::1 回环，映射的访问会被直接拒绝
    host: "0.0.0.0",
    port: 3000,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      // 多页：主应用 + 标注工具（车轮标注台，独立原生页面）
      input: {
        main: resolve(__dirname, "index.html"),
        "wheel-labeler": resolve(__dirname, "wheel-labeler/index.html"),
      },
    },
  },
});
