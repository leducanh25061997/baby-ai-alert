"""Test alert_policy — logic timing thuần (không cần cv2/mediapipe/telegram).

Chạy: python tests/test_alert_policy.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alert_policy import (
    Watchdog, HeartbeatPolicy, CalibrationReminder, CalibrationConditionWarner,
)


# ============== Watchdog ==============

def test_watchdog_alert_only_after_degrade_sec():
    """Sự cố < degrade_sec → chưa báo; >= → báo 1 lần (không lặp)."""
    w = Watchdog(degrade_sec=20)
    issue = {'too_dark': 'tối'}
    # t=0..19: chưa báo
    for t in range(0, 20):
        assert w.step(t, issue) == []
    # t=20: báo 1 lần
    ev = w.step(20, issue)
    assert len(ev) == 1 and ev[0].kind == 'alert' and ev[0].name == 'too_dark'
    # t=21..40: không báo lại
    for t in range(21, 40):
        assert w.step(t, issue) == []
    print("✅ test_watchdog_alert_only_after_degrade_sec")


def test_watchdog_recover_after_alert():
    """Sự cố đã báo rồi hết → phát sự kiện 'recover' đúng 1 lần."""
    w = Watchdog(degrade_sec=10)
    issue = {'low_fps': 'chậm'}
    for t in range(0, 10):
        w.step(t, issue)
    assert w.step(10, issue)[0].kind == 'alert'
    # hết sự cố
    ev = w.step(11, {})
    assert len(ev) == 1 and ev[0].kind == 'recover' and ev[0].name == 'low_fps'
    # tiếp tục hết → không phát recover lần nữa
    assert w.step(12, {}) == []
    print("✅ test_watchdog_recover_after_alert")


def test_watchdog_no_recover_if_never_alerted():
    """Sự cố thoáng qua (< degrade_sec) rồi hết → KHÔNG báo gì cả."""
    w = Watchdog(degrade_sec=20)
    w.step(0, {'too_dark': 'x'})
    w.step(5, {'too_dark': 'x'})
    ev = w.step(6, {})           # hết sớm, chưa từng alert
    assert ev == []
    print("✅ test_watchdog_no_recover_if_never_alerted")


# ============== HeartbeatPolicy ==============

def test_heartbeat_disabled():
    hb = HeartbeatPolicy(interval_sec=0)
    for t in range(0, 100):
        assert hb.step(t) is False
    print("✅ test_heartbeat_disabled")


def test_heartbeat_fires_on_interval():
    """Lần đầu chỉ đặt mốc; sau đó fire mỗi interval."""
    hb = HeartbeatPolicy(interval_sec=60)
    assert hb.step(1000.0) is False        # đặt mốc, không fire ngay
    assert hb.step(1059.0) is False
    assert hb.step(1060.0) is True
    assert hb.step(1061.0) is False
    assert hb.step(1120.0) is True
    print("✅ test_heartbeat_fires_on_interval")


# ============== CalibrationReminder ==============

def test_calib_reminder_disabled():
    r = CalibrationReminder(enabled=False, remind_sec=60)
    for t in range(0, 200):
        assert r.step(t, calib_done=False, face_present=False) is False
    print("✅ test_calib_reminder_disabled")


def test_calib_reminder_reminds_when_no_face():
    """Chưa calib + không thấy mặt → nhắc sau mỗi remind_sec."""
    r = CalibrationReminder(enabled=True, remind_sec=60)
    fired = []
    t = 0.0
    while t < 130.0:
        if r.step(t, calib_done=False, face_present=False):
            fired.append(round(t, 1))
        t += 1.0
    # đặt mốc ở t=0, nhắc tại 60 và 120
    assert fired == [60.0, 120.0], f"got {fired}"
    print(f"✅ test_calib_reminder_reminds_when_no_face  ({fired})")


def test_calib_reminder_silent_when_face_or_done():
    """Thấy mặt hoặc đã calib xong → không nhắc + reset mốc."""
    r = CalibrationReminder(enabled=True, remind_sec=60)
    # 70s không thấy mặt
    for t in range(0, 70):
        r.step(t, calib_done=False, face_present=False)
    # thấy mặt → reset, không nhắc
    assert r.step(70, calib_done=False, face_present=True) is False
    # mất mặt lại từ t=71 → phải đếm lại từ đầu (nhắc ở 71+60=131)
    fired = []
    for t in range(71, 140):
        if r.step(t, calib_done=False, face_present=False):
            fired.append(t)
    assert fired == [131], f"Phải đếm lại từ lúc mất mặt, got {fired}"
    # đã calib xong → không bao giờ nhắc
    assert r.step(200, calib_done=True, face_present=False) is False
    print("✅ test_calib_reminder_silent_when_face_or_done")


# ============== CalibrationConditionWarner ==============

def test_calib_warner_disabled():
    """enabled=False → không bao giờ nhắc dù điều kiện kém kéo dài."""
    w = CalibrationConditionWarner(enabled=False, grace_sec=4, remind_sec=60)
    for t in range(0, 200):
        assert w.step(t, calibrating=True, conditions_bad=True) is False
    print("✅ test_calib_warner_disabled")


def test_calib_warner_silent_when_conditions_good():
    """Điều kiện tốt → không nhắc."""
    w = CalibrationConditionWarner(enabled=True, grace_sec=4, remind_sec=60)
    for t in range(0, 200):
        assert w.step(t, calibrating=True, conditions_bad=False) is False
    print("✅ test_calib_warner_silent_when_conditions_good")


def test_calib_warner_silent_when_not_calibrating():
    """Không trong pha hiệu chỉnh → không nhắc dù điều kiện kém."""
    w = CalibrationConditionWarner(enabled=True, grace_sec=4, remind_sec=60)
    for t in range(0, 200):
        assert w.step(t, calibrating=False, conditions_bad=True) is False
    print("✅ test_calib_warner_silent_when_not_calibrating")


def test_calib_warner_warns_after_grace_then_repeats():
    """Điều kiện kém liên tục → nhắc lần đầu sau grace (4s), lặp mỗi 60s."""
    w = CalibrationConditionWarner(enabled=True, grace_sec=4, remind_sec=60)
    fired = []
    t = 0.0
    while t < 130.0:
        if w.step(t, calibrating=True, conditions_bad=True):
            fired.append(round(t, 1))
        t += 1.0
    # mốc xấu đặt ở t=0; nhắc đầu tại t=4 (đủ grace), rồi 64, 124
    assert fired == [4.0, 64.0, 124.0], f"got {fired}"
    print(f"✅ test_calib_warner_warns_after_grace_then_repeats  ({fired})")


def test_calib_warner_resets_when_conditions_recover():
    """Kém một lúc (chưa quá grace) rồi tốt lại → quên; kém lại phải chờ grace từ đầu."""
    w = CalibrationConditionWarner(enabled=True, grace_sec=4, remind_sec=60)
    # kém 2s (chưa đủ grace)
    assert w.step(0, calibrating=True, conditions_bad=True) is False
    assert w.step(2, calibrating=True, conditions_bad=True) is False
    # tốt lại tại t=3 → reset mốc
    assert w.step(3, calibrating=True, conditions_bad=False) is False
    # kém lại từ t=5 → mốc đặt lại ở 5, nhắc đầu tại 5+4=9
    fired = []
    t = 5.0
    while t < 12.0:
        if w.step(t, calibrating=True, conditions_bad=True):
            fired.append(round(t, 1))
        t += 1.0
    assert fired == [9.0], f"Phải chờ lại grace từ lúc kém lại, got {fired}"
    print("✅ test_calib_warner_resets_when_conditions_recover")


# ============== runner ==============

if __name__ == "__main__":
    tests = [
        test_watchdog_alert_only_after_degrade_sec,
        test_watchdog_recover_after_alert,
        test_watchdog_no_recover_if_never_alerted,
        test_heartbeat_disabled,
        test_heartbeat_fires_on_interval,
        test_calib_reminder_disabled,
        test_calib_reminder_reminds_when_no_face,
        test_calib_reminder_silent_when_face_or_done,
        test_calib_warner_disabled,
        test_calib_warner_silent_when_conditions_good,
        test_calib_warner_silent_when_not_calibrating,
        test_calib_warner_warns_after_grace_then_repeats,
        test_calib_warner_resets_when_conditions_recover,
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
