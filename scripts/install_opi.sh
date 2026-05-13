#!/bin/bash
# One-shot installer cho Orange Pi (3B/5/5B/5 Plus).
#
# Xử lý 3 vấn đề kinh điển khi cài project trên OPi:
#   1. /tmp là tmpfs RAM ~1GB → đầy khi pip cài torch (419MB) → "No space left"
#      Fix: set TMPDIR = $HOME/tmp (disk thật)
#   2. torch>=2.4 pull nvidia_cudnn_cu13 (433MB) dù chạy CPU-only
#      Fix: pin torch<2.4 trong requirements.txt + cleanup nvidia-* sau cài
#   3. pip cache cũ có thể chứa wheel hỏng / nvidia bloat từ lần trước
#      Fix: pip cache purge + --no-cache-dir
#
# Idempotent — chạy lại nhiều lần OK. Tự động tạo venv nếu chưa có.

set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

echo "=========================================="
echo "Baby AI Alert — Orange Pi installer"
echo "Project dir: $PROJ_ROOT"
echo "=========================================="

# --- Kiểm tra disk + RAM trước khi bắt đầu ---
echo ""
echo "=== Kiểm tra space ==="
df -h "$HOME" | tail -1 | awk '{print "  HOME free :", $4, "("$5" used)"}'
df -h /tmp  | tail -1 | awk '{print "  /tmp free :", $4, "("$5" used, type ="$1")"}'
free -h | grep -E "^(Mem|Swap)" | awk '{print "  "$1, "free :", $4}'

# --- Bước 1: tạo TMPDIR trên disk thật ---
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"
echo ""
echo "✅ TMPDIR=$TMPDIR (tránh /tmp tmpfs đầy)"

# --- Bước 2: venv ---
if [ ! -d "$PROJ_ROOT/venv" ]; then
    echo ""
    echo "=== Tạo virtualenv ==="
    python3 -m venv venv
fi
source "$PROJ_ROOT/venv/bin/activate"
echo "✅ Venv: $PROJ_ROOT/venv (Python $(python --version 2>&1 | awk '{print $2}'))"

# --- Bước 3: pip mới ---
echo ""
echo "=== Upgrade pip + wheel ==="
python -m pip install --upgrade pip wheel setuptools --no-cache-dir

# --- Bước 4: dọn dẹp installation cũ (nếu có) ---
echo ""
echo "=== Cleanup torch + nvidia-* + cuda-* cũ (nếu có) ==="
python -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
nvidia_pkgs=$(python -m pip list 2>/dev/null | grep -Ei "^(nvidia-|cuda-toolkit|triton)" | awk '{print $1}' || true)
if [ -n "$nvidia_pkgs" ]; then
    echo "Xoá: $nvidia_pkgs"
    echo "$nvidia_pkgs" | xargs python -m pip uninstall -y || true
else
    echo "(không có nvidia/cuda package — tốt)"
fi

# --- Bước 5: clear pip cache ---
echo ""
echo "=== Clear pip cache (free disk) ==="
python -m pip cache purge || true

# --- Bước 6: cài torch CPU-only TRƯỚC (chặn ultralytics pull torch 2.4+) ---
echo ""
echo "=== Cài torch<2.4 CPU-only TRƯỚC ==="
echo "(torch>=2.4 pull về nvidia_cudnn_cu13 ~433MB — project không cần)"
python -m pip install --no-cache-dir 'torch>=2.0,<2.4'

# --- Bước 7: cài rest từ requirements.txt ---
echo ""
echo "=== Cài requirements.txt còn lại ==="
python -m pip install --no-cache-dir -r requirements.txt

# --- Bước 8: chạy fix_env.sh để verify numpy<2 + opencv<4.11 ---
echo ""
echo "=== Verify numpy/opencv version + clean nvidia còn sót ==="
bash "$PROJ_ROOT/scripts/fix_env.sh"

# --- Bước 9: smoke test import ---
echo ""
echo "=== Smoke test imports ==="
python -c "
import sys
errors = []
for pkg in ('cv2', 'mediapipe', 'numpy', 'telegram', 'ultralytics', 'torch'):
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', '?')
        print(f'  ✅ {pkg:<15} {v}')
    except Exception as e:
        print(f'  ❌ {pkg:<15} {e}')
        errors.append(pkg)
sys.exit(1 if errors else 0)
"

# --- Final check: không còn nvidia bloat ---
echo ""
echo "=== Final check: nvidia/cuda package (PHẢI rỗng) ==="
remaining=$(python -m pip list 2>/dev/null | grep -Ei "^(nvidia-|cuda-toolkit|triton)" || true)
if [ -n "$remaining" ]; then
    echo "⚠️  Vẫn còn nvidia-* package:"
    echo "$remaining"
    echo "→ Chạy: pip list | grep -Ei 'nvidia|cuda' | awk '{print \$1}' | xargs pip uninstall -y"
else
    echo "✅ Sạch — không có nvidia/cuda package nào"
fi

echo ""
echo "=========================================="
echo "✅ CÀI XONG. Chạy:"
echo "   cd $PROJ_ROOT"
echo "   source venv/bin/activate"
echo "   python src/main.py"
echo "=========================================="
