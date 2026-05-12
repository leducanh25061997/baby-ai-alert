#!/bin/bash
# Fix môi trường numpy/opencv triệt để.
#
# Vấn đề:
#   - mediapipe 0.10.x  compile chống NumPy 1.x → cần numpy<2
#   - opencv-python(-contrib)(-headless) >=4.11 compile chống NumPy 2.x → cần <4.11
#   - Có 4 variant opencv: opencv-python, opencv-contrib-python,
#     opencv-python-headless, opencv-contrib-python-headless — tất cả share
#     cv2 namespace, cài chung sẽ xung đột.
#
# Action:
#   1. Uninstall TẤT CẢ 4 variant opencv (clean slate)
#   2. Cài lại opencv-python<4.11 + numpy<2
#
# Idempotent.

set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$PROJ_ROOT/venv/bin/activate" ]; then
    source "$PROJ_ROOT/venv/bin/activate"
    echo "✅ Activated venv: $PROJ_ROOT/venv"
fi

echo ""
echo "=== Phiên bản hiện tại ==="
python -c "import numpy; print('numpy   :', numpy.__version__)" 2>/dev/null || echo "numpy   : (chưa cài)"
python -c "import cv2;   print('cv2     :', cv2.__version__)" 2>/dev/null || echo "cv2     : (chưa cài)"

echo ""
echo "=== Các opencv package đang cài ==="
python -m pip list 2>/dev/null | grep -i "opencv" || echo "(không có)"

echo ""
echo "=== Bước 1: Uninstall tất cả opencv variant để clean slate ==="
for pkg in opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless; do
    python -m pip uninstall -y "$pkg" 2>/dev/null || true
done

echo ""
echo "=== Bước 2: Cài numpy<2 + opencv-python<4.11 ==="
python -m pip install 'numpy<2' 'opencv-python<4.11' --force-reinstall

echo ""
echo "=== Sau khi fix ==="
python -c "import numpy; print('numpy   :', numpy.__version__)"
python -c "import cv2;   print('cv2     :', cv2.__version__)"
python -m pip list 2>/dev/null | grep -i "opencv"

echo ""
echo "✅ Xong. Chạy: python src/main.py"
