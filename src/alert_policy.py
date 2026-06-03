"""Logic TIMING/QUYẾT ĐỊNH thuần — KHÔNG phụ thuộc cv2/mediapipe/telegram.

Tách khỏi main.py để unit-test độc lập (giống state_machine.py). main.py chỉ
còn việc gọi các policy này rồi thực thi side-effect (gửi Telegram, in log).

Gồm 4 policy nhỏ:
  - MotionAbsencePolicy : khi nào bắn cảnh báo "nghi bất động/ngưng thở".
  - Watchdog           : khi nào báo "giám sát suy giảm" / "đã khôi phục".
  - HeartbeatPolicy    : khi nào gửi heartbeat "vẫn đang canh".
  - CalibrationReminder: khi nào nhắc "đưa mặt vào khung" (chưa hiệu chỉnh được).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ---------------- Motion-absence (nghi bất động) ----------------

@dataclass
class AbsenceAlert:
    elapsed: float      # đã vắng cử động bao lâu
    is_first: bool      # True = cảnh báo đầu của đợt; False = re-alert


class MotionAbsencePolicy:
    """Quyết định bắn cảnh báo khi VẮNG cử động liên tục >= absent_sec, lặp lại
    mỗi repeat_sec. Chỉ tính khi có trẻ trong khung (subject_present) và đã
    calibrate ngưỡng chuyển động (ready)."""

    def __init__(self, absent_sec: float, repeat_sec: float):
        self.absent_sec = absent_sec      # <= 0 → TẮT
        self.repeat_sec = repeat_sec
        self.last_motion_t: Optional[float] = None
        self.last_alert_at: Optional[float] = None

    def reset(self):
        self.last_motion_t = None
        self.last_alert_at = None

    def step(self, now: float, subject_present: bool, has_motion: bool,
             ready: bool) -> Optional[AbsenceAlert]:
        if self.absent_sec <= 0 or not ready:
            return None
        if not subject_present:                 # không có trẻ → không tính bất động
            self.last_motion_t = now
            self.last_alert_at = None
            return None
        if has_motion:                          # còn cử động → reset đồng hồ
            self.last_motion_t = now
            self.last_alert_at = None
            return None
        if self.last_motion_t is None:
            self.last_motion_t = now
            return None
        elapsed = now - self.last_motion_t
        if elapsed < self.absent_sec:
            return None
        if self.last_alert_at is None:
            self.last_alert_at = now
            return AbsenceAlert(elapsed, True)
        if now - self.last_alert_at >= self.repeat_sec:
            self.last_alert_at = now
            return AbsenceAlert(elapsed, False)
        return None


# ---------------- Watchdog (tự giám sát) ----------------

@dataclass
class HealthEvent:
    kind: str        # 'alert' (bắt đầu suy giảm) | 'recover' (đã khôi phục)
    name: str        # 'camera_frozen' | 'too_dark' | 'low_fps' | ...
    message: str


class Watchdog:
    """Báo 'suy giảm' khi một sự cố kéo dài >= degrade_sec (mỗi sự cố báo 1 lần),
    và báo 'đã khôi phục' khi sự cố hết. Chống spam bằng tập đã-thông-báo."""

    def __init__(self, degrade_sec: float):
        self.degrade_sec = degrade_sec
        self._since: dict[str, float] = {}
        self._notified: set[str] = set()

    def reset(self):
        self._since.clear()
        self._notified.clear()

    def step(self, now: float, issues: dict[str, str]) -> list[HealthEvent]:
        """issues: {name: message} các sự cố ĐANG gặp ở frame này."""
        events: list[HealthEvent] = []
        for name, msg in issues.items():
            self._since.setdefault(name, now)
            if (now - self._since[name] >= self.degrade_sec
                    and name not in self._notified):
                self._notified.add(name)
                events.append(HealthEvent('alert', name, msg))
        for name in list(self._since.keys()):
            if name not in issues:
                if name in self._notified:
                    events.append(HealthEvent('recover', name, ''))
                    self._notified.discard(name)
                del self._since[name]
        return events


# ---------------- Heartbeat ----------------

class HeartbeatPolicy:
    """Trả True khi tới hạn gửi heartbeat (mỗi interval_sec). interval<=0 → TẮT.
    Lần gọi đầu chỉ đặt mốc (không gửi ngay lúc khởi động)."""

    def __init__(self, interval_sec: float):
        self.interval_sec = interval_sec
        self._last: Optional[float] = None

    def step(self, now: float) -> bool:
        if self.interval_sec <= 0:
            return False
        if self._last is None:
            self._last = now
            return False
        if now - self._last >= self.interval_sec:
            self._last = now
            return True
        return False


# ---------------- Calibration reminder ----------------

class CalibrationReminder:
    """Trả True khi cần nhắc 'đưa mặt vào khung' — chỉ khi CHƯA hiệu chỉnh xong
    VÀ không thấy mặt; lặp mỗi remind_sec. Thấy mặt → reset mốc nhắc."""

    def __init__(self, enabled: bool, remind_sec: float):
        self.enabled = enabled
        self.remind_sec = remind_sec
        self._last: Optional[float] = None

    def reset(self):
        self._last = None

    def step(self, now: float, calib_done: bool, face_present: bool) -> bool:
        if not self.enabled or calib_done or face_present:
            if face_present:
                self._last = None
            return False
        if self._last is None:
            self._last = now
            return False
        if now - self._last >= self.remind_sec:
            self._last = now
            return True
        return False
