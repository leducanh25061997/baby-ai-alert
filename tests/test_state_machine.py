"""Test state machine. Không có dependency ngoài stdlib.

Chạy: python tests/test_state_machine.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from state_machine import (
    OcclusionStateMachine,
    STATE_ALERT, STATE_SAFE, STATE_NO_FACE,
    TRIGGER_HISTOGRAM, TRIGGER_FACE_LOST,
)


# Helper: chạy nhiều frame liên tiếp và trả về kết quả cuối
def run_seq(fsm, sequence):
    last = None
    for kwargs in sequence:
        last = fsm.step(**kwargs)
    return last


def test_safe_flow():
    """Mặt thấy bình thường → SAFE, không alert."""
    fsm = OcclusionStateMachine()
    r = fsm.step(now=10.0, face_present=True, occluded_by_histogram=False)
    assert r.state == STATE_SAFE, f"got {r.state}"
    assert not r.should_alert
    assert r.elapsed == 0.0
    print("✅ test_safe_flow")


def test_histogram_alert_fires_at_threshold():
    """Mũi/miệng bị che 15s liên tục → alert."""
    fsm = OcclusionStateMachine(threshold_sec=15)
    # t=0: bắt đầu bị che
    r = fsm.step(now=0.0, face_present=True, occluded_by_histogram=True)
    assert r.state == STATE_ALERT
    assert r.trigger == TRIGGER_HISTOGRAM
    assert not r.should_alert  # mới 0s, chưa đủ 15s
    # t=14.9: gần đủ ngưỡng
    r = fsm.step(now=14.9, face_present=True, occluded_by_histogram=True)
    assert not r.should_alert
    # t=15.0: đủ ngưỡng → bắn alert
    r = fsm.step(now=15.0, face_present=True, occluded_by_histogram=True)
    assert r.should_alert, "Phải bắn alert tại t=15s"
    assert r.trigger == TRIGGER_HISTOGRAM
    # t=15.1: alert đã bắn, không bắn lại
    r = fsm.step(now=15.1, face_present=True, occluded_by_histogram=True)
    assert not r.should_alert, "Không được bắn lại"
    print("✅ test_histogram_alert_fires_at_threshold")


def test_histogram_recovery_resets():
    """Bị che rồi hết bị che → reset, lần sau lại đếm từ đầu."""
    fsm = OcclusionStateMachine(threshold_sec=15)
    fsm.step(now=0.0, face_present=True, occluded_by_histogram=True)
    fsm.step(now=10.0, face_present=True, occluded_by_histogram=True)
    # Recovery
    r = fsm.step(now=10.5, face_present=True, occluded_by_histogram=False)
    assert r.state == STATE_SAFE
    assert fsm.occlusion_start is None
    # Bị che lại từ đầu
    r = fsm.step(now=11.0, face_present=True, occluded_by_histogram=True)
    assert r.state == STATE_ALERT
    assert r.elapsed < 1.0  # đếm từ 11.0, không phải 0.0
    print("✅ test_histogram_recovery_resets")


def test_face_lost_critical_bug_fix():
    """KỊCH BẢN NGUY HIỂM NHẤT: mặt bị phủ kín → face_mesh không thấy mặt nữa.

    Trước fix: code reset bộ đếm khi không thấy mặt → KHÔNG BAO GIỜ alert.
    Sau fix:  state machine vẫn đếm và bắn alert đúng 15s.
    """
    fsm = OcclusionStateMachine(threshold_sec=15, grace_sec=1.5)

    # t=0: thấy mặt bình thường
    r = fsm.step(now=0.0, face_present=True, occluded_by_histogram=False)
    assert r.state == STATE_SAFE

    # t=0.5: chăn phủ kín → face_mesh mất tracking
    r = fsm.step(now=0.5, face_present=False, occluded_by_histogram=False)
    assert r.state == STATE_ALERT, "Phải nghi ngờ ngay khi vừa mất mặt"
    assert r.trigger == TRIGGER_FACE_LOST

    # t=1..14: vẫn không thấy mặt, vẫn đếm
    for t in [2.0, 5.0, 10.0, 14.9]:
        r = fsm.step(now=t, face_present=False, occluded_by_histogram=False)
        assert r.state == STATE_ALERT
        assert not r.should_alert, f"Chưa đủ ngưỡng tại t={t}"

    # t=15.0: đủ ngưỡng → ALERT BẮN! (chính là cái bug nghiêm trọng đã fix)
    r = fsm.step(now=15.0, face_present=False, occluded_by_histogram=False)
    assert r.should_alert, "FIX BUG: phải bắn alert dù không thấy mặt!"
    assert r.trigger == TRIGGER_FACE_LOST
    print("✅ test_face_lost_critical_bug_fix")


def test_face_lost_briefly_then_returns_no_alert():
    """Mặt mất 0.5s rồi quay lại bình thường → không được bắn alert."""
    fsm = OcclusionStateMachine(threshold_sec=15)
    fsm.step(now=0.0, face_present=True, occluded_by_histogram=False)
    fsm.step(now=0.5, face_present=False, occluded_by_histogram=False)
    # Quay lại bình thường
    r = fsm.step(now=1.0, face_present=True, occluded_by_histogram=False)
    assert r.state == STATE_SAFE
    assert fsm.occlusion_start is None
    assert not r.should_alert
    print("✅ test_face_lost_briefly_then_returns_no_alert")


def test_face_gone_long_no_yolo_resets():
    """Mặt mất quá lâu, không có YOLO → coi như rời khung, reset (tránh false alert)."""
    fsm = OcclusionStateMachine(threshold_sec=15, gone_reset_sec=25)
    fsm.step(now=0.0, face_present=True, occluded_by_histogram=False)
    fsm.step(now=0.5, face_present=False, occluded_by_histogram=False)
    # Đến đây chưa quá gone_reset_sec, vẫn ALERT
    r = fsm.step(now=10.0, face_present=False, occluded_by_histogram=False)
    assert r.state == STATE_ALERT  # Vẫn còn trong khoảng đếm
    # NHƯNG vì đếm liên tục không có break, sau 15s nó sẽ alert
    # Kịch bản chuẩn: đến 15s đã alert, sau đó alert_sent=True nên không bắn nữa
    r = fsm.step(now=15.5, face_present=False, occluded_by_histogram=False)
    assert r.should_alert  # bắn 1 lần
    r = fsm.step(now=16.0, face_present=False, occluded_by_histogram=False)
    assert not r.should_alert  # không bắn nữa

    # Reset state machine để test scenario "không alert, chỉ rời khung"
    fsm2 = OcclusionStateMachine(threshold_sec=15, grace_sec=1.5, gone_reset_sec=25)
    # Mặt thấy 1 lần, sau đó biến mất nhưng lúc đó alert chưa kịp
    # Để alert không kịp, ta cần giả lập: lần đầu thấy mặt rất xa quá khứ
    # Thực tế: nếu mặt mất >25s liên tục thì sẽ reset
    fsm2.step(now=0.0, face_present=True, occluded_by_histogram=False)
    # Mất mặt ngay
    fsm2.step(now=1.0, face_present=False, occluded_by_histogram=False)
    # Skip qua mốc alert (15s) — alert đã bắn
    r = fsm2.step(now=20.0, face_present=False, occluded_by_histogram=False)
    assert fsm2.alert_sent  # đã bắn rồi
    # Sau gone_reset_sec (25s) — reset
    r = fsm2.step(now=30.0, face_present=False, occluded_by_histogram=False)
    assert r.state == STATE_NO_FACE
    assert fsm2.occlusion_start is None
    assert not fsm2.alert_sent
    print("✅ test_face_gone_long_no_yolo_resets")


def test_yolo_no_person_in_grace_still_alerts():
    """SAFETY-FIRST: YOLO=False trong grace KHÔNG được reset.

    Lý do thực tế trên Orange Pi: camera nhìn TOP-DOWN xuống mặt trẻ nằm,
    YOLO COCO 'person' class chủ yếu train trên người đứng/ngồi → top-down
    view của trẻ → YOLO HAY trả False sai. Khi giấy/chăn che kín mặt:
    MediaPipe mất tracking + YOLO trả False sai → nếu reset ngay sẽ MISS
    alert hoàn toàn (kịch bản nguy hiểm chết người).

    Vì vậy trong grace + đã nghi che → tin face_lost path, kệ YOLO.
    """
    fsm = OcclusionStateMachine(threshold_sec=15, grace_sec=1.5)
    fsm.step(now=0.0, face_present=True, occluded_by_histogram=False)
    # Mất mặt + YOLO=False trong grace → VẪN ALERT (safety-first)
    r = fsm.step(now=0.5, face_present=False, occluded_by_histogram=False,
                 person_in_frame=False)
    assert r.state == STATE_ALERT, "Trong grace phải đếm bất kể YOLO"
    assert r.trigger == TRIGGER_FACE_LOST
    # Sau grace, occlusion_start đã set → tiếp tục đếm dù YOLO=False
    r = fsm.step(now=10.0, face_present=False, occluded_by_histogram=False,
                 person_in_frame=False)
    assert r.state == STATE_ALERT
    assert not r.should_alert  # chưa đủ 15s
    # t=15.0 → đủ ngưỡng → BẮN bất kể YOLO=False
    r = fsm.step(now=15.0, face_present=False, occluded_by_histogram=False,
                 person_in_frame=False)
    assert r.should_alert, "Phải bắn alert dù YOLO=False — kịch bản giấy che mặt"
    assert r.trigger == TRIGGER_FACE_LOST
    print("✅ test_yolo_no_person_in_grace_still_alerts")


def test_yolo_no_person_no_prior_face_resets():
    """YOLO=False khi CHƯA TỪNG thấy mặt → trẻ chưa từng vào khung → reset.

    Đây là case 'cha mẹ chưa đặt trẻ vào nôi' — không nên alert.
    """
    fsm = OcclusionStateMachine(threshold_sec=15, grace_sec=1.5)
    # Chưa từng thấy mặt + YOLO=False → reset
    r = fsm.step(now=0.0, face_present=False, occluded_by_histogram=False,
                 person_in_frame=False)
    assert r.state == STATE_NO_FACE
    assert not r.should_alert
    assert fsm.occlusion_start is None
    # Suốt 30s không ai → vẫn NO_FACE
    r = fsm.step(now=30.0, face_present=False, occluded_by_histogram=False,
                 person_in_frame=False)
    assert r.state == STATE_NO_FACE
    assert not r.should_alert
    print("✅ test_yolo_no_person_no_prior_face_resets")


def test_yolo_person_present_keeps_counting_beyond_grace():
    """YOLO khẳng định CÓ người dù không thấy mặt → nghi bị phủ → vẫn đếm."""
    fsm = OcclusionStateMachine(threshold_sec=15, grace_sec=1.5, gone_reset_sec=25)

    # Thấy mặt rồi mất từ rất sớm — mất mặt suốt thời gian dài
    fsm.step(now=0.0, face_present=True, occluded_by_histogram=False)

    # Mất mặt liên tục — nhưng YOLO khẳng định CÓ người
    for t in [0.5, 5.0, 10.0, 14.9]:
        r = fsm.step(now=t, face_present=False, occluded_by_histogram=False,
                     person_in_frame=True)
        assert r.state == STATE_ALERT, f"YOLO thấy người, phải đếm tại t={t}"

    # t=15.0 — alert bắn
    r = fsm.step(now=15.0, face_present=False, occluded_by_histogram=False,
                 person_in_frame=True)
    assert r.should_alert
    print("✅ test_yolo_person_present_keeps_counting_beyond_grace")


def test_alert_sent_flag_prevents_spam():
    """Khi đã bắn alert 1 lần, không được bắn lại liên tục mỗi frame."""
    fsm = OcclusionStateMachine(threshold_sec=15)
    fsm.step(now=0.0, face_present=True, occluded_by_histogram=True)
    # Bắn 1 lần ở t=15
    r = fsm.step(now=15.0, face_present=True, occluded_by_histogram=True)
    assert r.should_alert
    # Suốt 60s sau, không bắn lại
    fired_count = 0
    for t in range(16, 75):
        r = fsm.step(now=float(t), face_present=True, occluded_by_histogram=True)
        if r.should_alert:
            fired_count += 1
    assert fired_count == 0, f"Bắn lại {fired_count} lần — sai!"
    print("✅ test_alert_sent_flag_prevents_spam")


def test_no_face_then_alert_then_recovery():
    """Full flow: NO_FACE → SAFE → ALERT (face_lost) → SAFE again."""
    fsm = OcclusionStateMachine(threshold_sec=15, grace_sec=1.5)

    # Chưa từng thấy mặt
    r = fsm.step(now=0.0, face_present=False, occluded_by_histogram=False)
    assert r.state == STATE_NO_FACE

    # Thấy mặt
    r = fsm.step(now=1.0, face_present=True, occluded_by_histogram=False)
    assert r.state == STATE_SAFE

    # Mất mặt
    r = fsm.step(now=1.5, face_present=False, occluded_by_histogram=False)
    assert r.state == STATE_ALERT

    # Mặt quay lại sạch
    r = fsm.step(now=2.0, face_present=True, occluded_by_histogram=False)
    assert r.state == STATE_SAFE
    assert not r.should_alert
    print("✅ test_no_face_then_alert_then_recovery")


def test_grace_period_starts_count_from_face_lost_moment():
    """occlusion_start phải được tính từ lúc mất mặt, không phải lúc step."""
    fsm = OcclusionStateMachine(threshold_sec=15, grace_sec=1.5)
    fsm.step(now=10.0, face_present=True, occluded_by_histogram=False)
    # 1 giây sau mất mặt
    fsm.step(now=11.0, face_present=False, occluded_by_histogram=False)
    # occlusion_start phải ≈ 10.0 (lúc mặt mất), không phải 11.0
    assert abs(fsm.occlusion_start - 10.0) < 0.01, \
        f"occlusion_start={fsm.occlusion_start}, kỳ vọng ≈10.0"
    # Vì vậy alert phải bắn ở t=25.0 (10.0 + 15), không phải 26.0
    r = fsm.step(now=24.9, face_present=False, occluded_by_histogram=False)
    assert not r.should_alert
    r = fsm.step(now=25.0, face_present=False, occluded_by_histogram=False)
    assert r.should_alert
    print("✅ test_grace_period_starts_count_from_face_lost_moment")


if __name__ == "__main__":
    tests = [
        test_safe_flow,
        test_histogram_alert_fires_at_threshold,
        test_histogram_recovery_resets,
        test_face_lost_critical_bug_fix,
        test_face_lost_briefly_then_returns_no_alert,
        test_face_gone_long_no_yolo_resets,
        test_yolo_no_person_in_grace_still_alerts,
        test_yolo_no_person_no_prior_face_resets,
        test_yolo_person_present_keeps_counting_beyond_grace,
        test_alert_sent_flag_prevents_spam,
        test_no_face_then_alert_then_recovery,
        test_grace_period_starts_count_from_face_lost_moment,
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
