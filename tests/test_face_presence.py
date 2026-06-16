"""Test FSM face-presence (thuần stdlib — không cv2/mediapipe/telegram).

Chạy: python tests/test_face_presence.py

Phủ 18 kịch bản bắt buộc cho thông báo face-presence. Mọi mốc thời gian là
`now` tường minh → mô phỏng monotonic clock, không phụ thuộc đồng hồ hệ thống.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from face_presence import (
    FacePresenceFSM, FacePresenceConfig, FacePresenceEvent,
    is_valid_detection, count_valid_faces,
    SingleFlight, OnceGuard,
    STATE_INITIALIZING, STATE_NO_FACE, STATE_FACE_CANDIDATE,
    STATE_FACE_CONFIRMED, STATE_NOTIFIED, STATE_WAITING_FOR_FACE_EXIT,
)


def cfg(**kw):
    """Config nhanh cho test — ngưỡng nhỏ để mô phỏng gọn."""
    base = dict(min_confidence=0.6, min_area_ratio=0.02,
                confirm_frames=3, confirm_sec=0.3,
                exit_frames=3, exit_sec=0.5, cooldown_sec=5.0)
    base.update(kw)
    return FacePresenceConfig(**base)


def drive(fsm, seq):
    """Chạy list (now, count[, ready]) → trả list event đã phát."""
    events = []
    for item in seq:
        if len(item) == 3:
            now, count, ready = item
        else:
            now, count = item
            ready = True
        ev = fsm.step(now, count, ready=ready)
        if ev is not None:
            assert isinstance(ev, FacePresenceEvent)
            events.append(ev)
    return events


# ---- 1. Khởi động không mặt: không event ----
def test_startup_no_face_no_event():
    fsm = FacePresenceFSM(cfg())
    seq = [(t * 0.1, 0) for t in range(50)]
    events = drive(fsm, seq)
    assert events == [], events
    assert fsm.state == STATE_NO_FACE
    print("✅ 1 test_startup_no_face_no_event")


# ---- 2. Detection thoáng qua: không confirm ----
def test_transient_detection_not_confirmed():
    fsm = FacePresenceFSM(cfg())
    # thấy mặt 2 lần rồi biến mất hẳn (chưa đủ confirm_frames=3 + confirm_sec=0.3)
    seq = [(0.0, 1), (0.1, 1)] + [(0.2 + t * 0.1, 0) for t in range(20)]
    events = drive(fsm, seq)
    assert events == [], events
    assert fsm.state == STATE_NO_FACE
    print("✅ 2 test_transient_detection_not_confirmed")


# ---- 3. Mặt ổn định đủ ngưỡng: đúng một event ----
def test_stable_face_one_event():
    fsm = FacePresenceFSM(cfg())
    seq = [(t * 0.1, 1) for t in range(10)]
    events = drive(fsm, seq)
    assert len(events) == 1, events
    assert fsm.state in (STATE_NOTIFIED, STATE_WAITING_FOR_FACE_EXIT)
    print(f"✅ 3 test_stable_face_one_event ({events[0].id})")


# ---- 4. Mặt đứng nguyên: không event thứ hai ----
def test_face_stays_no_second_event():
    fsm = FacePresenceFSM(cfg())
    seq = [(t * 0.1, 1) for t in range(200)]   # 20s đứng nguyên
    events = drive(fsm, seq)
    assert len(events) == 1, f"Đứng nguyên phải chỉ 1 event, got {len(events)}"
    print("✅ 4 test_face_stays_no_second_event")


# ---- 5. Miss vài frame: không re-arm ----
def test_brief_miss_no_rearm():
    fsm = FacePresenceFSM(cfg())
    # confirm xong, rồi miss 2 frame (exit_frames=3 → chưa đủ), lại thấy mặt
    seq = [(t * 0.1, 1) for t in range(10)]          # confirm + notified
    seq += [(1.0, 0), (1.1, 0)]                       # miss 2 frame (chưa exit)
    seq += [(1.2 + t * 0.1, 1) for t in range(10)]    # mặt lại
    events = drive(fsm, seq)
    assert len(events) == 1, f"Miss vài frame KHÔNG được tạo event mới, got {len(events)}"
    assert fsm.state == STATE_WAITING_FOR_FACE_EXIT, fsm.state
    print("✅ 5 test_brief_miss_no_rearm")


# ---- 6. Tất cả mặt rời đủ lâu: re-arm ----
def test_all_faces_leave_rearm():
    fsm = FacePresenceFSM(cfg())
    seq = [(t * 0.1, 1) for t in range(10)]          # notified
    # vắng đủ exit_frames=3 + exit_sec=0.5
    seq += [(1.0 + t * 0.2, 0) for t in range(6)]     # tới 2.0s vắng
    drive(fsm, seq)
    assert fsm.state == STATE_NO_FACE, fsm.state
    print("✅ 6 test_all_faces_leave_rearm")


# ---- 7. Rời rồi trở lại sau cooldown: event thứ hai ----
def test_return_after_cooldown_second_event():
    fsm = FacePresenceFSM(cfg(cooldown_sec=5.0))
    seq = [(t * 0.1, 1) for t in range(10)]          # event#1 ~ t=0.3
    seq += [(1.0 + t * 0.2, 0) for t in range(8)]     # rời ổn định
    # quay lại SAU cooldown (event#1 ~0.3 → cần now ≥ 5.3)
    seq += [(6.0 + t * 0.1, 1) for t in range(10)]
    events = drive(fsm, seq)
    assert len(events) == 2, f"Phải có event thứ hai sau cooldown, got {len(events)}"
    assert events[0].id != events[1].id
    print("✅ 7 test_return_after_cooldown_second_event")


# ---- 8. Trở lại trước cooldown: chưa thông báo ----
def test_return_before_cooldown_no_event():
    fsm = FacePresenceFSM(cfg(cooldown_sec=5.0))
    seq = [(t * 0.1, 1) for t in range(10)]          # event#1 ~ t=0.3
    seq += [(1.0 + t * 0.2, 0) for t in range(6)]     # rời ổn định (~2.0s)
    # quay lại TRƯỚC cooldown (now < 5.3)
    seq += [(2.5 + t * 0.1, 1) for t in range(10)]    # tới ~3.4s
    events = drive(fsm, seq)
    assert len(events) == 1, f"Trước cooldown KHÔNG được thông báo, got {len(events)}"
    assert fsm.state == STATE_FACE_CONFIRMED, fsm.state
    print("✅ 8 test_return_before_cooldown_no_event")


# ---- 9. Nhiều mặt: một event ----
def test_multiple_faces_one_event():
    fsm = FacePresenceFSM(cfg())
    seq = [(t * 0.1, 3) for t in range(10)]
    events = drive(fsm, seq)
    assert len(events) == 1, f"Nhiều mặt vẫn chỉ 1 event, got {len(events)}"
    print("✅ 9 test_multiple_faces_one_event")


# ---- 10. Một trong nhiều mặt còn lại: chưa re-arm ----
def test_one_of_many_remains_no_rearm():
    fsm = FacePresenceFSM(cfg())
    seq = [(t * 0.1, 3) for t in range(10)]          # notified với 3 mặt
    seq += [(1.0 + t * 0.1, 1) for t in range(30)]    # 2 mặt rời, còn 1 mặt
    events = drive(fsm, seq)
    assert len(events) == 1, events
    assert fsm.state == STATE_WAITING_FOR_FACE_EXIT, fsm.state
    print("✅ 10 test_one_of_many_remains_no_rearm")


# ---- 11. Camera unavailable: không crash, không face event ----
def test_camera_unavailable_no_event():
    fsm = FacePresenceFSM(cfg())
    # ready=False suốt (camera không mở được) — dù có "đếm" cũng phải freeze.
    seq = [(t * 0.1, 5, False) for t in range(50)]
    events = drive(fsm, seq)
    assert events == [], events
    assert fsm.state == STATE_INITIALIZING, fsm.state
    print("✅ 11 test_camera_unavailable_no_event")


# ---- 12. Permission denied: không crash ----
def test_permission_denied_no_event():
    # Mô phỏng: detector luôn báo lỗi → caller truyền ready=False.
    fsm = FacePresenceFSM(cfg())
    for t in range(30):
        ev = fsm.step(t * 0.1, 0, ready=False)
        assert ev is None
    assert fsm.state == STATE_INITIALIZING
    print("✅ 12 test_permission_denied_no_event")


# ---- 13. Camera reconnect: không tạo event giả ----
def test_camera_reconnect_no_fake_event():
    fsm = FacePresenceFSM(cfg())
    seq = [(t * 0.1, 1) for t in range(10)]          # event#1, đang WAITING
    # camera rớt: caller FREEZE (ready=False) nhiều step (count bừa cũng kệ)
    seq += [(1.0 + t * 0.1, 0, False) for t in range(20)]
    # camera lại, mặt vẫn còn (chưa từng exit) → KHÔNG được phát event mới
    seq += [(3.0 + t * 0.1, 1) for t in range(20)]
    events = drive(fsm, seq)
    assert len(events) == 1, f"Reconnect KHÔNG được tạo event giả, got {len(events)}"
    assert fsm.state == STATE_WAITING_FOR_FACE_EXIT, fsm.state
    print("✅ 13 test_camera_reconnect_no_fake_event")


# ---- 14. Telegram retry: không tạo duplicate FSM event ----
def test_telegram_retry_no_duplicate_fsm_event():
    fsm = FacePresenceFSM(cfg())
    seq = [(t * 0.1, 1) for t in range(5)]
    events = drive(fsm, seq)
    assert len(events) == 1
    ev = events[0]
    # caller retry telegram dùng CÙNG ev.id — FSM tiếp tục step nhưng KHÔNG phát mới
    more = drive(fsm, [(0.5 + t * 0.1, 1) for t in range(20)])
    assert more == [], "FSM không được tạo event mới mỗi frame khi mặt vẫn còn"
    # id ổn định, không đổi
    assert ev.id == "face-presence-1"
    print("✅ 14 test_telegram_retry_no_duplicate_fsm_event")


# ---- 15. Dispose nhiều lần: không exception (OnceGuard idempotent) ----
def test_dispose_multiple_times_no_exception():
    calls = []
    g = OnceGuard()
    assert g.run(lambda: calls.append(1)) is True
    assert g.run(lambda: calls.append(2)) is False   # lần 2: không chạy lại
    assert g.run(lambda: calls.append(3)) is False
    assert calls == [1], calls
    assert g.done is True
    print("✅ 15 test_dispose_multiple_times_no_exception")


# ---- 16. Camera được release đúng cách (đúng MỘT lần) ----
def test_camera_released_exactly_once():
    released = {"n": 0}
    def _release():
        released["n"] += 1
    g = OnceGuard()
    # mô phỏng release gọi từ nhiều đường thoát (finally + signal)
    g.run(_release)
    g.run(_release)
    assert released["n"] == 1, released
    print("✅ 16 test_camera_released_exactly_once")


# ---- 17. Analysis đang bận: không chạy phiên AI thứ hai ----
def test_single_flight_blocks_second_session():
    sf = SingleFlight()
    assert sf.acquire() is True       # phiên 1 chiếm
    assert sf.acquire() is False      # phiên 2 bị chặn (bỏ frame, không queue)
    sf.release()
    assert sf.acquire() is True       # sau khi xong mới chạy phiên mới
    print("✅ 17 test_single_flight_blocks_second_session")


# ---- 18. Monotonic clock: không bị ảnh hưởng khi đồng hồ hệ thống thay đổi ----
def test_monotonic_offset_invariant():
    """Cùng chuỗi delta nhưng offset gốc khác nhau → kết quả y hệt (FSM chỉ dùng
    hiệu các `now`, không dùng giá trị tuyệt đối)."""
    def run_with_offset(offset):
        fsm = FacePresenceFSM(cfg())
        seq = [(offset + t * 0.1, 1) for t in range(10)]
        return drive(fsm, seq), fsm.state
    ev_a, st_a = run_with_offset(0.0)
    ev_b, st_b = run_with_offset(1_000_000.0)   # đồng hồ lớn bất kỳ
    assert len(ev_a) == len(ev_b) == 1
    assert st_a == st_b
    print("✅ 18 test_monotonic_offset_invariant")


# ---- bonus: lọc detection hình học ----
def test_is_valid_detection_rules():
    c = cfg(min_confidence=0.6, min_area_ratio=0.02)
    # hợp lệ: bbox giữa frame, đủ to, conf cao
    assert is_valid_detection((0.3, 0.3, 0.2, 0.2), 0.9, c)
    # confidence thấp
    assert not is_valid_detection((0.3, 0.3, 0.2, 0.2), 0.4, c)
    # quá nhỏ (area 0.01 < 0.02)
    assert not is_valid_detection((0.3, 0.3, 0.1, 0.1), 0.9, c)
    # w/h âm
    assert not is_valid_detection((0.3, 0.3, -0.2, 0.2), 0.9, c)
    # tràn biên bất thường
    assert not is_valid_detection((0.95, 0.3, 0.3, 0.3), 0.9, c)
    assert not is_valid_detection((-0.2, 0.3, 0.3, 0.3), 0.9, c)
    # count_valid_faces: None → 0; mix
    assert count_valid_faces(None, c) == 0
    assert count_valid_faces([((0.3, 0.3, 0.2, 0.2), 0.9),
                              ((0.0, 0.0, 0.05, 0.05), 0.9)], c) == 1
    print("✅ bonus test_is_valid_detection_rules")


if __name__ == "__main__":
    tests = [
        test_startup_no_face_no_event,
        test_transient_detection_not_confirmed,
        test_stable_face_one_event,
        test_face_stays_no_second_event,
        test_brief_miss_no_rearm,
        test_all_faces_leave_rearm,
        test_return_after_cooldown_second_event,
        test_return_before_cooldown_no_event,
        test_multiple_faces_one_event,
        test_one_of_many_remains_no_rearm,
        test_camera_unavailable_no_event,
        test_permission_denied_no_event,
        test_camera_reconnect_no_fake_event,
        test_telegram_retry_no_duplicate_fsm_event,
        test_dispose_multiple_times_no_exception,
        test_camera_released_exactly_once,
        test_single_flight_blocks_second_session,
        test_monotonic_offset_invariant,
        test_is_valid_detection_rules,
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
