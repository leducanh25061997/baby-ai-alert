"""Test OcclusionDetector (đã đơn giản hóa: chỉ skin_ratio + histogram).

Trọng tâm: vật che thật (chăn/gối/khăn) → occluded; mặt sạch (kể cả dao động
nhẹ về sáng/màu) → KHÔNG occluded. Đây chính là lớp lỗi báo nhầm cũ.

Chạy: python tests/test_occlusion_detector.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import cv2

from occlusion_detector import (
    OcclusionDetector, compute_signals,
    NOSE_TIP, MOUTH_CENTER, MIN_CALIB_SAMPLES,
)


# ===================== helpers =====================

class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def make_landmarks(nose=(0.5, 0.40), mouth=(0.5, 0.55), n_indices=20):
    lms = [FakeLandmark(0.0, 0.0) for _ in range(n_indices)]
    lms[NOSE_TIP]     = FakeLandmark(*nose)
    lms[MOUTH_CENTER] = FakeLandmark(*mouth)
    lms[13] = FakeLandmark(mouth[0], mouth[1] - 0.025)
    lms[17] = FakeLandmark(mouth[0], mouth[1] + 0.025)
    return lms


def face_frame(seed=0, w=320, h=240, skin_jitter=0):
    rng = np.random.default_rng(seed)
    f = np.full((h, w, 3), 35, dtype=np.uint8)
    face = np.full((140, 140, 3), 0, dtype=np.uint8)
    face[:] = (140, 170, 210)
    if skin_jitter > 0:
        noise = rng.integers(-skin_jitter, skin_jitter + 1, face.shape, dtype=np.int16)
        face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.line(face, (70, 30),  (70, 60), (90, 110, 150), 1)
    cv2.line(face, (50, 80),  (90, 80), (60, 80, 130),  1)
    cv2.line(face, (55, 90),  (85, 90), (60, 80, 130),  1)
    cv2.circle(face, (50, 50), 3, (0, 0, 0), -1)
    cv2.circle(face, (90, 50), 3, (0, 0, 0), -1)
    f[50:190, 90:230] = face
    return f


def blanket_frame(w=320, h=240, color=(100, 100, 100)):
    return np.full((h, w, 3), color, dtype=np.uint8)


def red_blanket_frame(w=320, h=240):
    """Chăn đỏ saturate — hue gần skin nhưng S/V khác → vẫn phải occluded."""
    return np.full((h, w, 3), (50, 50, 200), dtype=np.uint8)


def _calibrate(det, frame, lms):
    h, w = frame.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(frame, lms, w, h)
    return det.finalize_calibration()


# ===================== tests =====================

def test_compute_signals_basic():
    patch = cv2.resize(face_frame()[50:190, 90:230], (50, 50))
    sigs = compute_signals(patch)
    assert {'hist', 'skin_ratio'} == set(sigs.keys())
    assert sigs['hist'].shape == (36, 32)
    assert 0.0 <= sigs['skin_ratio'] <= 1.0
    assert sigs['skin_ratio'] > 0.3, f"skin={sigs['skin_ratio']:.2f}"
    print(f"✅ test_compute_signals_basic  (skin={sigs['skin_ratio']:.2f})")


def test_blanket_has_low_skin():
    patch = cv2.resize(blanket_frame()[50:100, 50:100], (50, 50))
    sigs = compute_signals(patch)
    assert sigs['skin_ratio'] < 0.05
    print(f"✅ test_blanket_has_low_skin  (skin={sigs['skin_ratio']:.2f})")


def test_calibration_needs_enough_samples():
    det = OcclusionDetector()
    lms = make_landmarks()
    f = face_frame()
    h, w = f.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES - 5):
        det.add_calibration_sample(f, lms, w, h)
    ok, msg = det.finalize_calibration()
    assert not ok and not det.is_ready
    print(f"✅ test_calibration_needs_enough_samples  ({msg[:40]})")


def test_calibration_succeeds_on_stable_face():
    det = OcclusionDetector()
    ok, msg = _calibrate(det, face_frame(), make_landmarks())
    assert ok, msg
    assert det.is_ready
    assert det.calibration_quality > 0.8, f"q={det.calibration_quality:.2f}"
    print(f"✅ test_calibration_succeeds_on_stable_face  (q={det.calibration_quality:.2f})")


def test_calibration_fails_on_blanket_landmark():
    det = OcclusionDetector()
    ok, msg = _calibrate(det, blanket_frame(), make_landmarks())
    assert not ok and "skin" in msg.lower()
    print(f"✅ test_calibration_fails_on_blanket_landmark  ({msg[:50]}...)")


def test_check_on_same_frame_safe():
    det = OcclusionDetector()
    lms = make_landmarks()
    f = face_frame()
    ok, _ = _calibrate(det, f, lms)
    assert ok
    h, w = f.shape[:2]
    r = det.check(f, lms, w, h, prev_in_alert=False)
    assert r is not None and not r.occluded, (
        f"same frame phải KHÔNG occluded; nose={r.nose_votes_for_occluded}/2 "
        f"mouth={r.mouth_votes_for_occluded}/2")
    assert r.nose_hist_corr > 0.95 and r.mouth_hist_corr > 0.95
    print(f"✅ test_check_on_same_frame_safe  (votes={r.total_votes}/4)")


def test_check_on_blanket_alerts():
    det = OcclusionDetector()
    lms = make_landmarks()
    f = face_frame()
    ok, _ = _calibrate(det, f, lms)
    assert ok
    h, w = f.shape[:2]
    r = det.check(blanket_frame(w=w, h=h), lms, w, h, prev_in_alert=False)
    assert r is not None and r.occluded, (
        f"chăn phải occluded; nose={r.nose_votes_for_occluded}/2 "
        f"mouth={r.mouth_votes_for_occluded}/2")
    print(f"✅ test_check_on_blanket_alerts  "
          f"(nose={r.nose_votes_for_occluded}/2 mouth={r.mouth_votes_for_occluded}/2)")


def test_check_on_red_blanket_alerts():
    det = OcclusionDetector()
    lms = make_landmarks()
    f = face_frame()
    ok, _ = _calibrate(det, f, lms)
    assert ok
    h, w = f.shape[:2]
    r = det.check(red_blanket_frame(w=w, h=h), lms, w, h, prev_in_alert=False)
    assert r is not None and r.occluded, (
        f"chăn đỏ phải occluded; nose={r.nose_votes_for_occluded}/2 "
        f"mouth={r.mouth_votes_for_occluded}/2")
    print(f"✅ test_check_on_red_blanket_alerts  "
          f"(nose={r.nose_votes_for_occluded}/2 mouth={r.mouth_votes_for_occluded}/2)")


def test_clear_face_with_jitter_no_false_alert():
    """REGRESSION (lỗi cũ events/162626,162643): mặt sạch skin~1.0 nhưng tín
    hiệu phụ dao động vẫn bị báo che. Sau khi bỏ edge/lap_var: calibrate trên
    mặt, check trên mặt khác (cùng tone, nhiễu sáng/màu nhẹ) → KHÔNG occluded."""
    det = OcclusionDetector()
    lms = make_landmarks()
    ok, _ = _calibrate(det, face_frame(seed=0), lms)
    assert ok
    f2 = face_frame(seed=7, skin_jitter=20)   # mặt khác, vẫn là da
    h, w = f2.shape[:2]
    r = det.check(f2, lms, w, h, prev_in_alert=False)
    assert r is not None and not r.occluded, (
        f"❌ mặt sạch dao động nhẹ báo nhầm che:\n"
        f"   nose V={r.nose_votes_for_occluded}/2 hist={r.nose_hist_corr:.2f} skin={r.nose_skin_ratio:.2f}\n"
        f"   mouth V={r.mouth_votes_for_occluded}/2 hist={r.mouth_hist_corr:.2f} skin={r.mouth_skin_ratio:.2f}")
    print(f"✅ test_clear_face_with_jitter_no_false_alert  "
          f"(nose hist={r.nose_hist_corr:.2f} skin={r.nose_skin_ratio:.2f})")


def test_reset_clears_state():
    det = OcclusionDetector()
    _calibrate(det, face_frame(), make_landmarks())
    assert det.is_ready
    det.reset()
    assert not det.is_ready and det.nose is None and det.mouth is None
    print("✅ test_reset_clears_state")


def test_check_returns_none_before_ready():
    det = OcclusionDetector()
    f = face_frame()
    h, w = f.shape[:2]
    assert det.check(f, make_landmarks(), w, h, prev_in_alert=False) is None
    print("✅ test_check_returns_none_before_ready")


def test_baseline_does_not_update_in_alert():
    det = OcclusionDetector()
    lms = make_landmarks()
    f = face_frame()
    _calibrate(det, f, lms)
    saved = det.nose.hist.copy()
    h, w = f.shape[:2]
    for _ in range(20):
        det.check(f, lms, w, h, prev_in_alert=True)
    assert np.allclose(det.nose.hist, saved)
    print("✅ test_baseline_does_not_update_in_alert")


if __name__ == "__main__":
    tests = [
        test_compute_signals_basic,
        test_blanket_has_low_skin,
        test_calibration_needs_enough_samples,
        test_calibration_succeeds_on_stable_face,
        test_calibration_fails_on_blanket_landmark,
        test_check_on_same_frame_safe,
        test_check_on_blanket_alerts,
        test_check_on_red_blanket_alerts,
        test_clear_face_with_jitter_no_false_alert,
        test_reset_clears_state,
        test_check_returns_none_before_ready,
        test_baseline_does_not_update_in_alert,
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
