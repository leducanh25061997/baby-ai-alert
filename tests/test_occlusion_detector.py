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
    NOSE_TIP, MOUTH_CENTER, MOUTH_LANDMARK_INDICES,
    MIN_CALIB_SAMPLES,
)


# ===================== helpers =====================

class FakeLandmark:
    """Mimic mediapipe NormalizedLandmark — chỉ cần x, y normalized."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


def make_landmarks(nose=(0.5, 0.40), mouth=(0.5, 0.55), n_indices=20):
    """List landmark mô phỏng mediapipe.
       - NOSE_TIP (4)        — nose tip
       - 13                  — upper lip top (slightly above mouth center)
       - MOUTH_CENTER (14)   — mouth center
       - 17                  — lower lip bottom (slightly below mouth center)
    """
    lms = [FakeLandmark(0.0, 0.0) for _ in range(n_indices)]
    lms[NOSE_TIP]     = FakeLandmark(*nose)
    lms[MOUTH_CENTER] = FakeLandmark(*mouth)
    # Multi-landmark sampling: 13 (upper) và 17 (lower) đặt sát mouth_center
    lms[13] = FakeLandmark(mouth[0], mouth[1] - 0.025)
    lms[17] = FakeLandmark(mouth[0], mouth[1] + 0.025)
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
    """Frame toàn màu xám đồng nhất — không skin, không edge, không texture."""
    f = np.full((h, w, 3), color, dtype=np.uint8)
    return f


def red_blanket_frame(w=320, h=240):
    """Chăn đỏ saturate — có thể overlap với skin range, để test edge case."""
    f = np.full((h, w, 3), (50, 50, 200), dtype=np.uint8)  # BGR red
    return f


def hand_frame(w=320, h=240):
    """Tay back skin tone phủ kín — case BAY hardest:
       - Skin tone giống mặt → histogram + skin signal đều OK
       - Có ít texture (1-2 knuckle line) → edge density có thể qua threshold
       - Laplacian variance phải thấp hơn mặt (vì uniform hơn) → trigger
    """
    f = np.full((h, w, 3), 35, dtype=np.uint8)
    hand = np.full((140, 140, 3), (140, 170, 210), dtype=np.uint8)
    # 1 knuckle line nhẹ — ít chi tiết hơn nhiều so với mặt (môi, mũi, lông)
    cv2.line(hand, (40, 70), (100, 70), (135, 165, 205), 1)
    f[50:190, 90:230] = hand
    return f


def low_detail_face_frame(w=320, h=240):
    """Mặt thật adult skin mịn với detail rất thấp — mô phỏng patch rơi vào
    vùng cheek-skin có ít edge (real-world baseline edge~0.01-0.02, lap~10-30).

    Mục đích: reproduce bug user gặp với face_baseline thấp + absolute floor
    quá cao gây persistent false positive trên mặt sạch.
    """
    f = np.full((h, w, 3), 35, dtype=np.uint8)
    face = np.full((140, 140, 3), (140, 170, 210), dtype=np.uint8)
    # Chỉ 1 đường mảnh mô phỏng môi mờ — KHÔNG có nhiều detail như face_frame()
    cv2.line(face, (50, 80), (90, 80), (130, 160, 200), 1)
    f[50:190, 90:230] = face
    return f


# ===================== tests =====================

def test_compute_signals_basic():
    """compute_signals trả về dict đúng key + giá trị hợp lý (4 signal)."""
    patch = face_frame()[50:190, 90:230]
    patch = cv2.resize(patch, (50, 50))
    sigs = compute_signals(patch)
    assert {'hist', 'skin_ratio', 'edge_density', 'lap_var'} <= set(sigs.keys())
    assert sigs['hist'].shape == (36, 32)
    assert 0.0 <= sigs['skin_ratio'] <= 1.0
    assert 0.0 <= sigs['edge_density'] <= 1.0
    assert sigs['lap_var'] >= 0
    assert sigs['skin_ratio'] > 0.3, f"skin={sigs['skin_ratio']:.2f}, mong cao"
    print(f"✅ test_compute_signals_basic  (skin={sigs['skin_ratio']:.2f} "
          f"edge={sigs['edge_density']:.3f} lap={sigs['lap_var']:.0f})")


def test_blanket_has_low_skin_and_low_lap_var():
    """Chăn xám: skin ratio ≈ 0, edge ≈ 0, lap_var ≈ 0."""
    patch = blanket_frame()[50:100, 50:100]
    patch = cv2.resize(patch, (50, 50))
    sigs = compute_signals(patch)
    assert sigs['skin_ratio']   < 0.05
    assert sigs['edge_density'] < 0.05
    assert sigs['lap_var']      < 5.0, f"Blanket lap_var={sigs['lap_var']:.0f}"
    print(f"✅ test_blanket_has_low_skin_and_low_lap_var  "
          f"(skin={sigs['skin_ratio']:.2f} edge={sigs['edge_density']:.3f} "
          f"lap={sigs['lap_var']:.0f})")


def test_face_lap_var_higher_than_hand():
    """Face patch (lip area) phải có lap_var cao hơn hand patch — đây là core
    discriminator cho case tay che."""
    face_patch = face_frame()[107:157, 135:185]  # mouth area
    face_patch = cv2.resize(face_patch, (50, 50))
    hand_patch = hand_frame()[107:157, 135:185]
    hand_patch = cv2.resize(hand_patch, (50, 50))
    face_lap = compute_signals(face_patch)['lap_var']
    hand_lap = compute_signals(hand_patch)['lap_var']
    assert face_lap > hand_lap, (
        f"Face lap_var ({face_lap:.0f}) phải > hand lap_var ({hand_lap:.0f})"
    )
    ratio = face_lap / max(hand_lap, 1.0)
    assert ratio > 1.5, (
        f"Ratio face/hand={ratio:.2f} quá thấp — lap_var không discriminating"
    )
    print(f"✅ test_face_lap_var_higher_than_hand  "
          f"(face={face_lap:.0f} hand={hand_lap:.0f} ratio={ratio:.1f}x)")


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
          f"(votes={result.total_votes}/8 corr nose={result.nose_hist_corr:.2f} "
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
        f"got nose={result.nose_votes_for_occluded}/4 mouth={result.mouth_votes_for_occluded}/4"
    )
    assert result.nose_votes_for_occluded  >= 2
    assert result.mouth_votes_for_occluded >= 2
    print(f"✅ test_check_on_blanket_alerts  "
          f"(nose={result.nose_votes_for_occluded}/4 mouth={result.mouth_votes_for_occluded}/4)")


def test_check_on_red_blanket_still_alerts():
    """Edge case: chăn đỏ overlap với skin HSV range. Histogram + edge + lap_var bù lại."""
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
    assert result.occluded, (
        f"Chăn đỏ phải occluded dù skin range overlap, "
        f"got nose={result.nose_votes_for_occluded}/4 mouth={result.mouth_votes_for_occluded}/4"
    )
    print(f"✅ test_check_on_red_blanket_still_alerts  "
          f"(nose={result.nose_votes_for_occluded}/4 mouth={result.mouth_votes_for_occluded}/4)")


def test_check_on_hand_alerts():
    """KỊCH BẢN KHÓ NHẤT: tay che mặt — skin tone giống mặt nên histogram +
    skin vote KHÔNG. Phải dựa vào edge density + Laplacian variance để alert.

    Đây chính là bug user báo: tay che không cảnh báo. Test này verify đã fix.
    """
    det = OcclusionDetector()
    lms = make_landmarks()
    f_face = face_frame()
    h_, w_ = f_face.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(f_face, lms, w_, h_)
    ok, msg = det.finalize_calibration()
    assert ok, msg

    f_hand = hand_frame(w=w_, h=h_)
    result = det.check(f_hand, lms, w_, h_, prev_in_alert=False)
    assert result is not None
    assert result.occluded, (
        f"❌ Tay che mặt phải alert!\n"
        f"   nose votes={result.nose_votes_for_occluded}/4 "
        f"(hist={result.nose_hist_corr:.2f} skin={result.nose_skin_ratio:.2f} "
        f"edge={result.nose_edge_density:.3f} lap={result.nose_lap_var:.0f}, "
        f"baseline_lap={det.nose.lap_var:.0f} threshold={det.nose.lap_var_min:.0f})\n"
        f"   mouth votes={result.mouth_votes_for_occluded}/4 "
        f"(hist={result.mouth_hist_corr:.2f} skin={result.mouth_skin_ratio:.2f} "
        f"edge={result.mouth_edge_density:.3f} lap={result.mouth_lap_var:.0f}, "
        f"baseline_lap={det.mouth.lap_var:.0f} threshold={det.mouth.lap_var_min:.0f})"
    )
    print(f"✅ test_check_on_hand_alerts  "
          f"(nose={result.nose_votes_for_occluded}/4 mouth={result.mouth_votes_for_occluded}/4 | "
          f"face_lap nose={det.nose.lap_var:.0f} mouth={det.mouth.lap_var:.0f} | "
          f"hand_lap nose={result.nose_lap_var:.0f} mouth={result.mouth_lap_var:.0f})")


def test_blur_gate_suppresses_texture_votes():
    """BLUR GATE: khi cả khung mờ (texture_reliable=False) → bỏ phiếu edge/lap.

    Tay che (hand) bình thường vote nhờ edge+lap (texture). Khi báo 'đang mờ',
    2 phiếu này bị bỏ → hand không còn đủ vote → KHÔNG occluded. Đây là cách
    chống false-positive do autofocus hunting / motion blur (lỗi đã gặp trên Pi 4).
    """
    det = OcclusionDetector()
    lms = make_landmarks()
    f_face = face_frame()
    h_, w_ = f_face.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(f_face, lms, w_, h_)
    ok, msg = det.finalize_calibration()
    assert ok, msg

    f_hand = hand_frame(w=w_, h=h_)
    # texture đáng tin (mặc định) → hand vote occluded như cũ
    r_ok = det.check(f_hand, lms, w_, h_, prev_in_alert=False, texture_reliable=True)
    assert r_ok is not None and r_ok.occluded, "Texture tin cậy: hand phải occluded"
    # cả khung mờ → bỏ phiếu edge/lap → hand KHÔNG còn đủ vote
    r_blur = det.check(f_hand, lms, w_, h_, prev_in_alert=False, texture_reliable=False)
    assert r_blur is not None
    assert not r_blur.occluded, (
        f"Khi mờ phải bỏ phiếu texture → không occluded, "
        f"got nose={r_blur.nose_votes_for_occluded}/4 mouth={r_blur.mouth_votes_for_occluded}/4"
    )
    print(f"✅ test_blur_gate_suppresses_texture_votes  "
          f"(reliable: nose={r_ok.nose_votes_for_occluded}/4 mouth={r_ok.mouth_votes_for_occluded}/4 → "
          f"blurry: nose={r_blur.nose_votes_for_occluded}/4 mouth={r_blur.mouth_votes_for_occluded}/4)")


def test_low_detail_face_no_false_alert():
    """REGRESSION: face baseline thấp (real-world adult) + absolute floor cao
    → MỌI frame trên mặt sạch đều vote occluded → persistent false alert.

    Test này verify rằng face có baseline edge~0.01 / lap~10-30 phải KHÔNG
    bị nhầm thành occluded khi check trên frame y hệt.
    """
    det = OcclusionDetector()
    lms = make_landmarks()
    f = low_detail_face_frame()
    h_, w_ = f.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(f, lms, w_, h_)
    ok, msg = det.finalize_calibration()
    assert ok, msg
    # Check ngay frame y hệt → KHÔNG được alert
    result = det.check(f, lms, w_, h_, prev_in_alert=False)
    assert result is not None
    assert not result.occluded, (
        f"❌ Mặt sạch (low detail baseline) báo nhầm:\n"
        f"   nose V={result.nose_votes_for_occluded}/4 "
        f"(edge={result.nose_edge_density:.4f} thresh={det.nose.edge_min_density:.4f}, "
        f"lap={result.nose_lap_var:.1f} thresh={det.nose.lap_var_min:.1f})\n"
        f"   mouth V={result.mouth_votes_for_occluded}/4 "
        f"(edge={result.mouth_edge_density:.4f} thresh={det.mouth.edge_min_density:.4f}, "
        f"lap={result.mouth_lap_var:.1f} thresh={det.mouth.lap_var_min:.1f})"
    )
    print(f"✅ test_low_detail_face_no_false_alert  "
          f"(baseline mouth edge={det.mouth.edge_density:.3f} lap={det.mouth.lap_var:.1f} | "
          f"threshold edge={det.mouth.edge_min_density:.4f} lap={det.mouth.lap_var_min:.1f} | "
          f"votes={result.total_votes}/8)")


def test_hand_stability_under_landmark_drift():
    """Mô phỏng landmark drift (hand position thay đổi nhẹ giữa các frame) —
    detection PHẢI ổn định, không oscillate. Đây chính là bug user gặp:
    'đôi khi cảnh báo, không ổn định'.

    Multi-landmark sampling + MIN aggregation phải ensure tất cả frame đều vote
    occluded dù landmark có dịch.
    """
    det = OcclusionDetector()
    lms = make_landmarks()
    f_face = face_frame()
    h_, w_ = f_face.shape[:2]
    for _ in range(MIN_CALIB_SAMPLES + 5):
        det.add_calibration_sample(f_face, lms, w_, h_)
    ok, _ = det.finalize_calibration()
    assert ok

    f_hand = hand_frame(w=w_, h=h_)
    # Simulate landmark drift: jitter mouth landmark ±10px (≈ 0.03 normalized)
    n_occluded = 0
    n_total = 0
    drifts = [(0.0, 0.0), (0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02),
              (0.02, 0.02), (-0.02, -0.02), (0.0, 0.03), (0.03, 0.0)]
    for dx, dy in drifts:
        drifted = make_landmarks(
            nose=(0.5, 0.40), mouth=(0.5 + dx, 0.55 + dy)
        )
        r = det.check(f_hand, drifted, w_, h_, prev_in_alert=False)
        if r is None:
            continue
        n_total += 1
        if r.occluded:
            n_occluded += 1
    # Yêu cầu: ít nhất 90% frame phải vote occluded (ổn định cao)
    rate = n_occluded / max(n_total, 1)
    assert rate >= 0.9, (
        f"❌ Chỉ {n_occluded}/{n_total} frame vote occluded ({rate*100:.0f}%) — "
        f"chưa đủ ổn định. Mong ≥ 90%."
    )
    print(f"✅ test_hand_stability_under_landmark_drift  "
          f"({n_occluded}/{n_total} frame occluded = {rate*100:.0f}%)")


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
        test_blanket_has_low_skin_and_low_lap_var,
        test_face_lap_var_higher_than_hand,
        test_calibration_needs_enough_samples,
        test_calibration_succeeds_on_stable_face,
        test_calibration_fails_on_blanket_landmark,
        test_check_on_same_frame_safe,
        test_check_on_blanket_alerts,
        test_check_on_red_blanket_still_alerts,
        test_check_on_hand_alerts,
        test_blur_gate_suppresses_texture_votes,
        test_low_detail_face_no_false_alert,
        test_hand_stability_under_landmark_drift,
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
