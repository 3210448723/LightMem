#!/bin/bash
# LoCoMo 轮次分割可视化 - 本地 HTTP 服务启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p outputs

echo "=========================================="
echo "启动 HTTP 服务"
echo "==========================================" 
echo ""
echo "✅ 服务启动中..."
echo "📱 本机访问：http://localhost:8000"
echo "🌐 局域网访问：http://<你的IP地址>:8000"
echo "📄 页面文件：turn_segmentation_viewer.html"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

python3 -m http.server 8000 --bind 0.0.0.0