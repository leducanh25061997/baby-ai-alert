#!/bin/bash
# Fix môi trường: ép numpy<2 + opencv-python<4.11.
#
# Lý do:
#   - mediapipe 0.10.x compile chống NumPy 1.x → cần numpy<2
#   - opencv-python >=4.11 compile chống NumPy 2.x → cần opencv<4.11
#   Cả 2 phải downgrade cùng lúc, không thì conflict.
#
# Idempotent — chạy lại không hại.

set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Detect venv nếu có
if [ -f "$PROJ_ROOT/venv/bin/activate" ]; then
    source "$PROJ_ROOT/venv/bin/activate"
    echo "✅ Activated venv: $PROJ_ROOT/venv"
fi

echo ""
echo "=== Phiên bản hiện tại ==="
python -c "import numpy; print('numpy   :', numpy.__version__)" 2>/dev/null || echo "numpy   : (chưa cài)"
python -c "import cv2;   print('opencv  :', cv2.__version__)" 2>/dev/null || echo "opencv  : (chưa cài)"

NUMPY_OK=$(python -c "import numpy, sys; sys.exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null && echo "1" || echo "0")
OPENCV_OK=$(python -c "
import cv2, sys
parts = cv2.__version__.split('.')
major = int(parts[0]); minor = int(parts[1])
sys.exit(0 if (major == 4 and minor < 11) else 1)
" 2>/dev/null && echo "1" || echo "0")

if [ "$NUMPY_OK" = "1" ] && [ "$OPENCV_OK" = "1" ]; then
    echo ""
    echo "✅ numpy và opencv đã đúng version. Không cần fix."
    exit 0
fi

echo ""
echo "=== Bắt đầu fix ==="
echo "→ Uninstall opencv-contrib-python (nếu có, không cần thiết)..."
python -m pip uninstall -y opencv-contrib-python 2>/dev/null || true

echo ""
echo "→ Reinstall numpy<2 + opencv-python<4.11..."
python -m pip install 'numpy<2' 'opencv-python<4.11' --force-reinstall

echo ""
echo "=== Sau khi fix ==="
python -c "import numpy; print('numpy   :', numpy.__version__)"
python -c "import cv2;   print('opencv  :', cv2.__version__)"

echo ""
echo "✅ Xong. Chạy: python src/main.py"
