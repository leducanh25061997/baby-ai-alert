"""Test OcclusionDetector — yêu cầu numpy<2 + opencv-python được cài.

Tạo synthetic frames để verify:
  - Calibration với frame consistent → ready, quality cao
  - Calibration với skin% thấp → critical fail
  - Calibration với noisy patches → critical fail (high stddev)
  - Check với frame y hệt calibration → votes thấp, không occluded
  - Check với frame phủ kín (blanket) → votes cao, occluded
  - Reset → trở về not ready

Chạy: python tests/test_occlusion_detector.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import cv2

from occlusion_detector import (
    OcclusionDetector, CheckResult, compute_signals,
    NOSE_TIP, MOUTH_CENTER,
    MIN_CALIB_SAMPLES,
)


# ===================== helpers =====================

class FakeLandmark:
    """Mimic mediapipe NormalizedLandmark — chỉ cần x, y normalized."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


def make_landmarks(nose=(0.5, 0.40), mouth=(0.5, 0.55), n_indices=20):
    """List landmark, index NOSE_TIP/MOUTH_CENTER set, các index khác cũng có
    (mediapipe có 478 landmark; ta chỉ dùng 2 nên 20 là đủ)."""
    lms = [FakeLandmark(0.0, 0.0) for _ in range(n_indices)]
    lms[NOSE_TIP]     = FakeLandmark(*nose)
    lms[MOUTH_CENTER] = FakeLandmark(*mouth)
    return lms


def face_frame(seed=0, w=320, h=240, skin_jitter=0):
    """Tạo frame giả lập có 'mặt' với skin tone.
    seed → reproducible. skin_jitter → noise.
    """
    rng = np.random.default_rng(seed)
    f = np.full((h, w, 3), 35, dtype=np.uint8)  # tối, background

    # Face area: skin tone BGR ≈ (140, 170, 210) → HSV approx (10, 90, 210)
    face = np.full((140, 140, 3), 0, dtype=np.uint8)
    face[:] = (140, 170, 210)
    if skin_jitter > 0:
        noise = rng.integers(-skin_jitter, skin_jitter + 1, face.shape, dtype=np.int16)
        face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Edges (chi tiết mặt: mũi, môi)
    cv2.line(face, (70, 30),  (70, 60), (90, 110, 150), 1)  # sống mũi
    cv2.line(face, (50, 80),  (90, 80), (60, 80, 130),  1)  # đường môi
    cv2.line(face, (55, 90),  (85, 90), (60, 80, 130),  1)
    cv2.circle(face, (50, 50), 3, (0, 0, 0), -1)  # mắt trái
    cv2.circle(face, (90, 50), 3, (0, 0, 0), -1)  # mắt phải

    f[50:190, 90:230] = face
    return f


def blanket_frame(w=320, h=240, color=(100, 100, 100)):
    """Frame toàn màu xám đồng nhất — không skin, không edge."""
    f = np.full((h, w, 3), color, dtype=np.uint8)
    return f


def red_blanket_frame(w=320, h=240):
    """Chăn đỏ saturate — có thể overlap với skin range, để test edge case."""
    f = np.full((h, w, 3), (50, 50, 200), dtype=np.uint8)  # BGR red
    return f


# ===================== tests =====================

def test_compute_signals_basic():
    """compute_signals trả về dict đúng key + giá trị hợp lý."""
    patch = face_frame()[50:190, 90:230]
    patch = cv2.resize(patch, (50, 50))
    sigs = compute_signals(patch)
    assert 'hist' in sigs and 'skin_ratio' in sigs and 'edge_density' in sigs
    assert sigs['hist'].shape == (36, 32)
    assert 0.0 <= sigs['skin_ratio'] <= 1.0
    assert 0.0 <= sigs['edge_density'] <= 1.0
    # Face patch phải có skin_ratio cao
    assert sigs['skin_ratio'] > 0.3, f"skin={sigs['skin_ratio']:.2f}, mong cao"
    print(f"✅ test_compute_signals_basic  (skin={sigs['skin_ratio']:.2f} "
          f"edge={sigs['edge_density']:.3f})")


def test_blanket_has_low_skin():
    """Frame chăn đồng nhất phải có skin_ratio rất thấp."""
    patch = blanket_frame()[50:100, 50:100]
    patch = cv2.resize(patch, (50, 50))
    sigs = compute_signals(patch)
    assert sigs['skin_ratio'] < 0.05, \
        f"Blanket có skin={sigs['skin_ratio']:.2f} — mong gần 0"
    assert sigs['edge_density'] < 0.05, \
        f"Blanket có edge={sigs['edge_density']:.3f} — mong gần 0"
    print(f"✅ test_blanket_has_low_skin  (skin={sigs['skin_ratio']:.2f} "
          f"edge={sigs['edge_density']:.3f})")


def test_calibration_needs_enough_samples():
    """Calibrate < MIN_CALIB_SAMPLES → finalize fail."""
    det = OcclusionDetector()
    lms = make_landmarks()
    f = face_frame()
    h, w = f.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES - 5):
        det.add_calibration_sample(f, lms, w, h)
    ok, msg = det.finalize_calibration()
    assert not ok
    assert "Quá ít sample" in msg or "ít" in msg
    assert not det.is_ready
    print(f"✅ test_calibration_needs_enough_samples  ({msg[:50]})")


def test_calibration_succeeds_on_stable_face():
    """Calibrate với consistent face frame → ready, quality cao."""
    det = OcclusionDetector()
    lms = make_landmarks()
    h_, w_ = face_frame().shape[:2]
    # Cùng 1 frame nhiều lần
    f = face_frame()
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(f, lms, w_, h_)
    ok, msg = det.finalize_calibration()
    assert ok, f"Calibration FAIL: {msg}"
    assert det.is_ready
    assert det.calibration_quality > 0.8, \
        f"Quality={det.calibration_quality:.2f} mong > 0.8 (cùng 1 frame, phải rất stable)"
    print(f"✅ test_calibration_succeeds_on_stable_face  "
          f"(quality={det.calibration_quality:.2f})")


def test_calibration_fails_on_blanket_landmark():
    """Nếu landmark trỏ vào nơi không có skin (blanket frame) → critical fail."""
    det = OcclusionDetector()
    lms = make_landmarks()
    f = blanket_frame()
    h_, w_ = f.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(f, lms, w_, h_)
    ok, msg = det.finalize_calibration()
    assert not ok, "Phải fail vì skin% quá thấp"
    assert "skin" in msg.lower()
    assert not det.is_ready
    print(f"✅ test_calibration_fails_on_blanket_landmark  ({msg[:60]}...)")


def test_check_on_same_frame_safe():
    """Calibrate xong → check trên frame y hệt → KHÔNG occluded, votes thấp."""
    det = OcclusionDetector()
    lms = make_landmarks()
    f = face_frame()
    h_, w_ = f.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(f, lms, w_, h_)
    ok, _ = det.finalize_calibration()
    assert ok
    result = det.check(f, lms, w_, h_, prev_in_alert=False)
    assert result is not None
    assert not result.occluded, (
        f"Same frame phải KHÔNG occluded, "
        f"got votes nose={result.nose_votes_for_occluded}/3 mouth={result.mouth_votes_for_occluded}/3"
    )
    assert result.nose_hist_corr  > 0.95
    assert result.mouth_hist_corr > 0.95
    print(f"✅ test_check_on_same_frame_safe  "
          f"(votes={result.total_votes}/6 corr nose={result.nose_hist_corr:.2f} "
          f"mouth={result.mouth_hist_corr:.2f})")


def test_check_on_blanket_alerts():
    """Calibrate trên face, check trên blanket → occluded."""
    det = OcclusionDetector()
    lms = make_landmarks()
    f_face = face_frame()
    h_, w_ = f_face.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(f_face, lms, w_, h_)
    ok, _ = det.finalize_calibration()
    assert ok

    f_blanket = blanket_frame(w=w_, h=h_)
    result = det.check(f_blanket, lms, w_, h_, prev_in_alert=False)
    assert result is not None
    assert result.occluded, (
        f"Blanket phải occluded, "
        f"got votes nose={result.nose_votes_for_occluded}/3 mouth={result.mouth_votes_for_occluded}/3"
    )
    # Cả 2 patch nên có ≥2/3 vote
    assert result.nose_votes_for_occluded  >= 2
    assert result.mouth_votes_for_occluded >= 2
    print(f"✅ test_check_on_blanket_alerts  "
          f"(nose={result.nose_votes_for_occluded}/3 mouth={result.mouth_votes_for_occluded}/3)")


def test_check_on_red_blanket_still_alerts():
    """Edge case: chăn đỏ overlap với skin HSV range. Histogram + edge phải bù lại."""
    det = OcclusionDetector()
    lms = make_landmarks()
    f_face = face_frame()
    h_, w_ = f_face.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(f_face, lms, w_, h_)
    ok, _ = det.finalize_calibration()
    assert ok

    f_red = red_blanket_frame(w=w_, h=h_)
    result = det.check(f_red, lms, w_, h_, prev_in_alert=False)
    assert result is not None
    # Histogram chắc chắn rất khác, edge thấp → ít nhất 2/3 vote
    assert result.occluded, (
        f"Chăn đỏ phải occluded dù skin range overlap, "
        f"got votes nose={result.nose_votes_for_occluded}/3 mouth={result.mouth_votes_for_occluded}/3"
    )
    print(f"✅ test_check_on_red_blanket_still_alerts  "
          f"(nose={result.nose_votes_for_occluded}/3 mouth={result.mouth_votes_for_occluded}/3)")


def test_reset_clears_state():
    """reset() đưa detector về not ready, samples sạch."""
    det = OcclusionDetector()
    lms = make_landmarks()
    f = face_frame()
    h_, w_ = f.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(f, lms, w_, h_)
    det.finalize_calibration()
    assert det.is_ready
    det.reset()
    assert not det.is_ready
    assert det.nose is None and det.mouth is None
    assert det.calibration_quality == 0.0
    print("✅ test_reset_clears_state")


def test_check_returns_none_before_ready():
    """Check trước khi calibrate → None."""
    det = OcclusionDetector()
    lms = make_landmarks()
    f = face_frame()
    h_, w_ = f.shape[:2]
    assert det.check(f, lms, w_, h_, prev_in_alert=False) is None
    print("✅ test_check_returns_none_before_ready")


def test_baseline_does_not_update_in_alert():
    """Nếu prev_in_alert=True → baseline KHÔNG được update dù safe hiện tại."""
    det = OcclusionDetector()
    lms = make_landmarks()
    f = face_frame()
    h_, w_ = f.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(f, lms, w_, h_)
    det.finalize_calibration()
    saved_hist = det.nose.hist.copy()
    # Check với prev_in_alert=True → baseline không update
    for _ in range(20):
        det.check(f, lms, w_, h_, prev_in_alert=True)
    assert np.allclose(det.nose.hist, saved_hist), \
        "Baseline phải KHÔNG đổi khi prev_in_alert=True"
    print("✅ test_baseline_does_not_update_in_alert")


# ===================== runner =====================

if __name__ == "__main__":
    tests = [
        test_compute_signals_basic,
        test_blanket_has_low_skin,
        test_calibration_needs_enough_samples,
        test_calibration_succeeds_on_stable_face,
        test_calibration_fails_on_blanket_landmark,
        test_check_on_same_frame_safe,
        test_check_on_blanket_alerts,
        test_check_on_red_blanket_still_alerts,
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
