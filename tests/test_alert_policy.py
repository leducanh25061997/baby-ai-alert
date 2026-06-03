"""Test alert_policy — logic timing thuần (không cần cv2/mediapipe/telegram).

Chạy: python tests/test_alert_policy.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alert_policy import (
    MotionAbsencePolicy, Watchdog, HeartbeatPolicy, CalibrationReminder,
)


# ============== MotionAbsencePolicy ==============

def test_motion_absence_disabled():
    """absent_sec<=0 → không bao giờ bắn."""
    p = MotionAbsencePolicy(absent_sec=0, repeat_sec=15)
    for t in range(0, 100):
        assert p.step(t, subject_present=True, has_motion=False, ready=True) is None
    print("✅ test_motion_absence_disabled")


def test_motion_absence_not_ready():
    """Chưa calibrate ngưỡng (ready=False) → không bắn dù vắng cử động."""
    p = MotionAbsencePolicy(absent_sec=30, repeat_sec=15)
    for t in range(0, 100):
        assert p.step(t, subject_present=True, has_motion=False, ready=False) is None
    print("✅ test_motion_absence_not_ready")


def test_motion_absence_fires_after_threshold():
    """Vắng cử động đủ 30s (có trẻ, ready) → bắn cảnh báo đầu đúng mốc."""
    p = MotionAbsencePolicy(absent_sec=30, repeat_sec=15)
    fired = []
    t = 0.0
    while t < 35.0:
        d = p.step(t, subject_present=True, has_motion=False, ready=True)
        if d:
            fired.append(round(t, 1))
        t += 1.0
    assert fired == [30.0], f"Phải bắn 1 lần tại t=30, got {fired}"
    print("✅ test_motion_absence_fires_after_threshold")


def test_motion_absence_resets_on_motion():
    """Có cử động giữa chừng → reset đồng hồ; phải đếm lại 30s từ lúc đó."""
    p = MotionAbsencePolicy(absent_sec=30, repeat_sec=15)
    for t in range(0, 20):
        assert p.step(t, subject_present=True, has_motion=False, ready=True) is None
    # có cử động tại t=20 → reset đồng hồ về 20
    assert p.step(20, subject_present=True, has_motion=True, ready=True) is None
    fired = []
    for t in range(21, 52):
        d = p.step(t, subject_present=True, has_motion=False, ready=True)
        if d:
            fired.append(t)
    assert fired == [50], f"Reset ở t=20 → đủ 30 tại t=50, got {fired}"
    print("✅ test_motion_absence_resets_on_motion")


def test_motion_absence_resets_when_subject_absent():
    """Không có trẻ trong khung → không tính bất động (reset)."""
    p = MotionAbsencePolicy(absent_sec=30, repeat_sec=15)
    for t in range(0, 100):
        assert p.step(t, subject_present=False, has_motion=False, ready=True) is None
    print("✅ test_motion_absence_resets_when_subject_absent")


def test_motion_absence_re_alerts_every_repeat():
    """Sau cảnh báo đầu (t=30), re-alert mỗi 15s khi vẫn bất động."""
    p = MotionAbsencePolicy(absent_sec=30, repeat_sec=15)
    fired = []
    t = 0.0
    while t < 75.0:
        d = p.step(t, subject_present=True, has_motion=False, ready=True)
        if d:
            fired.append(round(t, 1))
        t += 0.5
    # đầu tại 30, re-alert 45, 60 (75 là biên)
    assert fired == [30.0, 45.0, 60.0], f"got {fired}"
    print(f"✅ test_motion_absence_re_alerts_every_repeat  ({fired})")


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


# ============== runner ==============

if __name__ == "__main__":
    tests = [
        test_motion_absence_disabled,
        test_motion_absence_not_ready,
        test_motion_absence_fires_after_threshold,
        test_motion_absence_resets_on_motion,
        test_motion_absence_resets_when_subject_absent,
        test_motion_absence_re_alerts_every_repeat,
        test_watchdog_alert_only_after_degrade_sec,
        test_watchdog_recover_after_alert,
        test_watchdog_no_recover_if_never_alerted,
        test_heartbeat_disabled,
        test_heartbeat_fires_on_interval,
        test_calib_reminder_disabled,
        test_calib_reminder_reminds_when_no_face,
        test_calib_reminder_silent_when_face_or_done,
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
