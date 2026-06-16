"""Test state machine ĐƠN GIẢN (che / mất người). Thuần stdlib.

Chạy: python tests/test_state_machine.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from state_machine import (
    OcclusionStateMachine,
    STATE_SAFE, STATE_COVERED, STATE_FACE_LOST, STATE_NO_PERSON,
    TRIGGER_COVERED, TRIGGER_FACE_LOST, TRIGGER_NO_PERSON,
)


def test_safe_flow():
    """Có người + không che → SAFE, không báo."""
    fsm = OcclusionStateMachine()
    r = fsm.step(now=10.0, person_present=True, covered=False)
    assert r.state == STATE_SAFE, r.state
    assert not r.should_alert
    assert r.elapsed == 0.0
    print("✅ test_safe_flow")


def test_covered_alert_fires_at_threshold():
    """Mũi/miệng bị che 15s liên tục → báo."""
    fsm = OcclusionStateMachine(threshold_sec=15)
    r = fsm.step(now=0.0, person_present=True, covered=True)
    assert r.state == STATE_COVERED
    assert r.trigger == TRIGGER_COVERED
    assert not r.should_alert
    r = fsm.step(now=14.9, person_present=True, covered=True)
    assert not r.should_alert
    r = fsm.step(now=15.0, person_present=True, covered=True)
    assert r.should_alert and r.trigger == TRIGGER_COVERED
    r = fsm.step(now=15.1, person_present=True, covered=True)
    assert not r.should_alert, "Không bắn lại trong cùng chu kỳ"
    print("✅ test_covered_alert_fires_at_threshold")


def test_covered_recovery_anti_flicker():
    """Bị che rồi hết che → reset SAU khi sạch đủ lâu (anti-flicker)."""
    fsm = OcclusionStateMachine(threshold_sec=15, safe_recovery_sec=1.5)
    fsm.step(now=0.0, person_present=True, covered=True)
    fsm.step(now=10.0, person_present=True, covered=True)
    r = fsm.step(now=10.5, person_present=True, covered=False)
    assert r.state == STATE_COVERED, "0.5s sạch chưa đủ để reset"
    assert fsm.covered_start is not None
    r = fsm.step(now=12.0, person_present=True, covered=False)
    assert r.state == STATE_SAFE
    assert fsm.covered_start is None
    print("✅ test_covered_recovery_anti_flicker")


def test_covered_repeats_every_threshold():
    """Còn bị che → báo lặp mỗi threshold_sec (4 lần trong 75s)."""
    fsm = OcclusionStateMachine(threshold_sec=15)
    fired = []
    t = 0.0
    while t < 75.0:
        r = fsm.step(now=t, person_present=True, covered=True)
        if r.should_alert:
            fired.append(round(t, 1))
        t += 0.1
    assert len(fired) == 4, f"Phải báo 4 lần; thực tế {fired}"
    for i in range(1, len(fired)):
        assert 14.9 < fired[i] - fired[i-1] < 15.2
    print("✅ test_covered_repeats_every_threshold")


def test_no_person_never_alerts():
    """Mất người: vào STATE_NO_PERSON + đếm elapsed để hiển thị, NHƯNG
    KHÔNG bao giờ gửi cảnh báo (đã bỏ logic cảnh báo mất người)."""
    fsm = OcclusionStateMachine(no_person_sec=15)
    r = fsm.step(now=0.0, person_present=False, covered=False)
    assert r.state == STATE_NO_PERSON
    assert r.trigger == TRIGGER_NO_PERSON
    assert not r.should_alert
    r = fsm.step(now=14.9, person_present=False, covered=False)
    assert not r.should_alert
    r = fsm.step(now=15.0, person_present=False, covered=False)
    assert not r.should_alert, "Mất người KHÔNG được báo"
    assert r.state == STATE_NO_PERSON
    assert abs(r.elapsed - 15.0) < 1e-9, "Vẫn đếm elapsed để hiển thị"
    print("✅ test_no_person_never_alerts")


def test_no_person_never_alerts_long():
    """Mất người kéo dài rất lâu → vẫn KHÔNG có cảnh báo nào."""
    fsm = OcclusionStateMachine(no_person_sec=15)
    fired = []
    t = 0.0
    while t < 120.0:
        r = fsm.step(now=t, person_present=False, covered=False)
        if r.should_alert:
            fired.append(round(t, 1))
        t += 0.1
    assert len(fired) == 0, f"Không được báo lần nào; thực tế {fired}"
    print("✅ test_no_person_never_alerts_long")


def test_person_returns_clears_no_person():
    """Đang đếm mất người, người quay lại → về SAFE ngay, dừng đếm."""
    fsm = OcclusionStateMachine(no_person_sec=15)
    fsm.step(now=0.0, person_present=False, covered=False)
    fsm.step(now=10.0, person_present=False, covered=False)
    r = fsm.step(now=11.0, person_present=True, covered=False)
    assert r.state == STATE_SAFE
    assert fsm.absent_start is None
    print("✅ test_person_returns_clears_no_person")


def test_lost_person_clears_covered():
    """Đang đếm che mà người biến mất → chuyển sang đếm 'mất người', bỏ che."""
    fsm = OcclusionStateMachine(threshold_sec=15, no_person_sec=15)
    fsm.step(now=0.0, person_present=True, covered=True)
    fsm.step(now=5.0, person_present=True, covered=True)
    r = fsm.step(now=6.0, person_present=False, covered=False)
    assert r.state == STATE_NO_PERSON
    assert fsm.covered_start is None, "Mất người phải xóa bộ đếm che"
    assert fsm.absent_start is not None
    print("✅ test_lost_person_clears_covered")


def test_covered_only_when_person_present():
    """covered=True nhưng không có người → vẫn coi là mất người (ưu tiên presence)."""
    fsm = OcclusionStateMachine()
    r = fsm.step(now=0.0, person_present=False, covered=True)
    assert r.state == STATE_NO_PERSON
    assert r.trigger == TRIGGER_NO_PERSON
    print("✅ test_covered_only_when_person_present")


def test_no_alert_before_threshold_then_safe():
    """Che ngắn <15s rồi hết → không báo, về SAFE sau khi sạch đủ lâu."""
    fsm = OcclusionStateMachine(threshold_sec=15, safe_recovery_sec=1.5)
    fsm.step(now=0.0, person_present=True, covered=True)
    r = fsm.step(now=2.0, person_present=True, covered=False)
    assert r.state == STATE_COVERED and not r.should_alert
    r = fsm.step(now=4.0, person_present=True, covered=False)
    assert r.state == STATE_SAFE
    assert not r.should_alert
    print("✅ test_no_alert_before_threshold_then_safe")


def test_face_lost_fires_at_threshold():
    """Có người (YOLO) nhưng MẤT MẶT 15s → báo FACE_LOST."""
    fsm = OcclusionStateMachine(face_lost_sec=15)
    r = fsm.step(now=0.0, person_present=True, covered=False, face_present=False)
    assert r.state == STATE_FACE_LOST and r.trigger == TRIGGER_FACE_LOST
    assert not r.should_alert
    r = fsm.step(now=14.9, person_present=True, covered=False, face_present=False)
    assert not r.should_alert
    r = fsm.step(now=15.0, person_present=True, covered=False, face_present=False)
    assert r.should_alert and r.trigger == TRIGGER_FACE_LOST
    r = fsm.step(now=15.1, person_present=True, covered=False, face_present=False)
    assert not r.should_alert, "Không bắn lại trong cùng chu kỳ"
    print("✅ test_face_lost_fires_at_threshold")


def test_face_present_is_safe():
    """Có người + thấy mặt + không che → SAFE (không kích nhánh mất mặt)."""
    fsm = OcclusionStateMachine(face_lost_sec=15)
    r = fsm.step(now=5.0, person_present=True, covered=False, face_present=True)
    assert r.state == STATE_SAFE and not r.should_alert
    print("✅ test_face_present_is_safe")


def test_face_lost_disabled():
    """face_lost_sec=0 → mất mặt KHÔNG bao giờ báo (về SAFE)."""
    fsm = OcclusionStateMachine(face_lost_sec=0)
    fired = []
    t = 0.0
    while t < 60.0:
        r = fsm.step(now=t, person_present=True, covered=False, face_present=False)
        if r.should_alert:
            fired.append(round(t, 1))
        assert r.state == STATE_SAFE, r.state
        t += 0.1
    assert len(fired) == 0, f"Tắt nhánh mất mặt mà vẫn báo: {fired}"
    print("✅ test_face_lost_disabled")


def test_face_lost_recovery_anti_flicker():
    """Mất mặt rồi mặt hiện lại → reset SAU khi thấy mặt đủ lâu (anti-flicker)."""
    fsm = OcclusionStateMachine(face_lost_sec=15, safe_recovery_sec=1.5)
    fsm.step(now=0.0, person_present=True, covered=False, face_present=False)
    fsm.step(now=10.0, person_present=True, covered=False, face_present=False)
    r = fsm.step(now=10.5, person_present=True, covered=False, face_present=True)
    assert r.state == STATE_FACE_LOST, "0.5s thấy mặt chưa đủ để reset"
    assert fsm.facelost_start is not None
    r = fsm.step(now=12.0, person_present=True, covered=False, face_present=True)
    assert r.state == STATE_SAFE
    assert fsm.facelost_start is None
    print("✅ test_face_lost_recovery_anti_flicker")


def test_covered_takes_priority_over_face_lost():
    """covered=True luôn ưu tiên hơn mất mặt (cùng là nguy cơ đường thở)."""
    fsm = OcclusionStateMachine(threshold_sec=10, face_lost_sec=15)
    r = fsm.step(now=0.0, person_present=True, covered=True, face_present=False)
    assert r.state == STATE_COVERED and r.trigger == TRIGGER_COVERED
    assert fsm.facelost_start is None
    print("✅ test_covered_takes_priority_over_face_lost")


def test_face_lost_needs_person():
    """Mất mặt nhưng cũng mất người → NO_PERSON (không báo mất mặt)."""
    fsm = OcclusionStateMachine(face_lost_sec=15)
    r = fsm.step(now=0.0, person_present=False, covered=False, face_present=False)
    assert r.state == STATE_NO_PERSON and not r.should_alert
    assert fsm.facelost_start is None
    print("✅ test_face_lost_needs_person")


def test_backward_compat_properties():
    """occlusion_start / last_alert_at vẫn dùng được cho UI countdown."""
    fsm = OcclusionStateMachine(threshold_sec=15)
    fsm.step(now=0.0, person_present=True, covered=True)
    assert abs(fsm.occlusion_start - 0.0) < 1e-9
    assert fsm.last_alert_at is None
    fsm.step(now=15.0, person_present=True, covered=True)
    assert fsm.last_alert_at is not None
    print("✅ test_backward_compat_properties")


if __name__ == "__main__":
    tests = [
        test_safe_flow,
        test_covered_alert_fires_at_threshold,
        test_covered_recovery_anti_flicker,
        test_covered_repeats_every_threshold,
        test_no_person_never_alerts,
        test_no_person_never_alerts_long,
        test_person_returns_clears_no_person,
        test_lost_person_clears_covered,
        test_covered_only_when_person_present,
        test_no_alert_before_threshold_then_safe,
        test_face_lost_fires_at_threshold,
        test_face_present_is_safe,
        test_face_lost_disabled,
        test_face_lost_recovery_anti_flicker,
        test_covered_takes_priority_over_face_lost,
        test_face_lost_needs_person,
        test_backward_compat_properties,
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
