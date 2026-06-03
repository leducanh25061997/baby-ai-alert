"""Test SceneMonitor — blur-gate, motion, frozen, luma. Cần numpy + opencv.

Chạy: python tests/test_scene_monitor.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import cv2

from scene_monitor import SceneMonitor


# ===================== helpers =====================

def sharp_frame(seed=0, w=320, h=240):
    """Frame nhiều chi tiết (high-frequency) → Laplacian variance cao."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def blurred_of(frame):
    """Làm mờ mạnh → Laplacian variance tụt → mô phỏng out-of-focus/motion blur."""
    return cv2.GaussianBlur(frame, (0, 0), sigmaX=8)


def gray_fill(val, w=320, h=240):
    return np.full((h, w, 3), val, dtype=np.uint8)


# ===================== tests =====================

def test_blur_gate_detects_global_blur():
    """Sau khi học baseline trên frame nét, frame mờ → is_blurry=True; nét lại → False."""
    sm = SceneMonitor(blur_drop_frac=0.45)
    sharp = sharp_frame()
    # Học baseline trên vài frame nét
    for _ in range(20):
        sm.analyze(sharp)
        sm.update_sharp_baseline()
    assert not sm.is_blurry, f"Frame nét không được coi là mờ (sharp={sm.sharpness:.0f}, base={sm.sharp_baseline:.0f})"
    base = sm.sharp_baseline
    # Frame mờ → sharpness tụt mạnh → blurry
    sm.analyze(blurred_of(sharp))
    assert sm.is_blurry, f"Frame mờ phải bị gate (sharp={sm.sharpness:.0f} < {0.45*base:.0f})"
    # Nét trở lại → hết blurry
    sm.analyze(sharp)
    assert not sm.is_blurry
    print(f"✅ test_blur_gate_detects_global_blur  "
          f"(baseline={base:.0f}, blurred_sharp={cv2.Laplacian(cv2.cvtColor(cv2.resize(blurred_of(sharp),(0,0),fx=0.25,fy=0.25),cv2.COLOR_BGR2GRAY),cv2.CV_64F).var():.0f})")


def test_blur_gate_off_before_baseline():
    """Chưa có baseline → is_blurry luôn False (warmup an toàn, không gate nhầm)."""
    sm = SceneMonitor()
    sm.analyze(blurred_of(sharp_frame()))
    assert not sm.is_blurry
    print("✅ test_blur_gate_off_before_baseline")


def test_frozen_frame_detection():
    """Frame y hệt nhau (diff=0) → is_frozen_frame=True; cảnh có thay đổi → False."""
    sm = SceneMonitor(frozen_eps=0.05)
    f = sharp_frame(seed=5)
    sm.analyze(f)
    sm.analyze(f.copy())          # y hệt → diff = 0
    assert sm.is_frozen_frame, f"Frame trùng phải là frozen (motion={sm.motion})"
    sm.analyze(sharp_frame(seed=6))   # khác hẳn
    assert not sm.is_frozen_frame
    print("✅ test_frozen_frame_detection")


def test_luma_dark_vs_bright():
    """Đo độ sáng trung bình: tối ~ thấp, sáng ~ cao."""
    sm = SceneMonitor()
    sm.analyze(gray_fill(10))
    assert sm.luma < 30, f"Frame tối luma={sm.luma:.0f}"
    sm.analyze(gray_fill(230))
    assert sm.luma > 200, f"Frame sáng luma={sm.luma:.0f}"
    print(f"✅ test_luma_dark_vs_bright")


def test_reset_clears_state():
    sm = SceneMonitor()
    sm.analyze(sharp_frame())
    sm.update_sharp_baseline()
    sm.reset()
    assert sm.sharp_baseline is None
    assert sm._prev_gray is None
    print("✅ test_reset_clears_state")


# ===================== runner =====================

if __name__ == "__main__":
    tests = [
        test_blur_gate_detects_global_blur,
        test_blur_gate_off_before_baseline,
        test_frozen_frame_detection,
        test_luma_dark_vs_bright,
        test_reset_clears_state,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"❌ {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
    total = len(tests)
    if failed == 0:
        print(f"\n🎉 {total}/{total} test PASS")
    else:
        print(f"\n❌ {failed}/{total} FAIL")
        sys.exit(1)
