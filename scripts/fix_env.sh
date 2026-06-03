#!/bin/bash
# Fix môi trường numpy/opencv/torch triệt để.
#
# Vấn đề 1 (numpy/opencv ABI):
#   - mediapipe 0.10.x  compile chống NumPy 1.x → cần numpy<2
#   - opencv-python(-contrib)(-headless) >=4.11 compile chống NumPy 2.x → cần <4.11
#   - 4 variant opencv share cv2 namespace, cài chung sẽ xung đột.
#
# Vấn đề 2 (CUDA bloat trên ARM CPU-only):
#   - torch>=2.4 trên Linux pull về nvidia-cudnn-cu13 (~433MB), nvidia-cublas-cu12,
#     cuda_toolkit... dù chạy CPU. Trên ARM /tmp tmpfs nhỏ → có thể fail "No space".
#   - Fix: pin torch<2.4 + uninstall mọi nvidia-* lỡ cài.
#
# Action:
#   1. Uninstall TẤT CẢ 4 variant opencv (clean slate)
#   2. Uninstall mọi nvidia-* / cuda-* package vô dụng trên CPU-only
#   3. Cài lại numpy<2 + opencv-python<4.11
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
python -c "import torch; print('torch   :', torch.__version__)" 2>/dev/null || echo "torch   : (chưa cài)"

echo ""
echo "=== Các opencv package đang cài ==="
python -m pip list 2>/dev/null | grep -i "opencv" || echo "(không có)"

echo ""
echo "=== Các nvidia/cuda package đang cài (NÊN không có) ==="
python -m pip list 2>/dev/null | grep -Ei "nvidia|cuda" || echo "(không có — tốt)"

echo ""
echo "=== Bước 1: Uninstall tất cả opencv variant để clean slate ==="
for pkg in opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless; do
    python -m pip uninstall -y "$pkg" 2>/dev/null || true
done

echo ""
echo "=== Bước 2: Uninstall mọi nvidia-* / cuda-* (project CPU-only) ==="
nvidia_pkgs=$(python -m pip list 2>/dev/null | grep -Ei "^(nvidia-|cuda-toolkit)" | awk '{print $1}')
if [ -n "$nvidia_pkgs" ]; then
    echo "Sẽ xoá: $nvidia_pkgs"
    echo "$nvidia_pkgs" | xargs python -m pip uninstall -y || true
else
    echo "(không có gì để xoá)"
fi

echo ""
echo "=== Bước 3: Cài numpy<2 + opencv-python<4.11 ==="
python -m pip install 'numpy<2' 'opencv-python<4.11' --force-reinstall --no-cache-dir

echo ""
echo "=== Sau khi fix ==="
python -c "import numpy; print('numpy   :', numpy.__version__)"
python -c "import cv2;   print('cv2     :', cv2.__version__)"
python -m pip list 2>/dev/null | grep -i "opencv"
echo ""
echo "nvidia/cuda package còn lại (phải rỗng):"
python -m pip list 2>/dev/null | grep -Ei "nvidia|cuda" || echo "(không có — sạch)"

echo ""
echo "✅ Xong. Chạy: python src/main.py"
