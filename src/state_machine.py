"""State machine ĐƠN GIẢN — chỉ 1 sự kiện cần BÁO:

  1. CHE mũi/miệng     : có người + vùng mũi/miệng bị che ≥ threshold → báo.

Trạng thái MẤT NGƯỜI (không thấy ai trong khung) vẫn được theo dõi để
log/UI hiển thị, NHƯNG không còn gửi cảnh báo (should_alert luôn False).

KHÔNG còn khái niệm "ngạt thở / nghi bị phủ kín".

Mỗi sự kiện: bắn lần đầu khi đủ ngưỡng thời gian, sau đó lặp lại mỗi
`repeat_sec` đến khi tình huống hết. Có anti-flicker để 1 frame nhiễu không
làm reset bộ đếm.

Module thuần stdlib → unit-test độc lập với cv2/mediapipe/telegram.
"""
from dataclasses import dataclass
from typing import Optional


# Trạng thái cấp UI
STATE_SAFE      = "SAFE"          # có người, mũi/miệng nhìn rõ
STATE_COVERED   = "COVERED"       # có người, mũi/miệng bị che (đang đếm/đã báo)
STATE_NO_PERSON = "NO_PERSON"     # không thấy ai trong khung

# Lý do trigger
TRIGGER_COVERED   = "covered"
TRIGGER_NO_PERSON = "no_person"


@dataclass
class StepResult:
    state: str
    elapsed: float          # giây đã ở trong tình huống hiện tại
    should_alert: bool      # frame này có cần gửi thông báo không
    trigger: str            # TRIGGER_COVERED | TRIGGER_NO_PERSON | ""


class OcclusionStateMachine:
    """Quản lý timing cho 2 cảnh báo độc lập.

    Đầu vào mỗi frame:
      - now            : timestamp (float seconds)
      - person_present : có thấy người/mặt trong khung không (đã smooth ở caller)
      - covered        : vùng mũi/miệng có bị che không (đã smooth ở caller)
    """

    def __init__(
        self,
        threshold_sec: float = 15.0,        # che bao lâu thì báo
        no_person_sec: float = 15.0,        # mất người bao lâu thì báo
        repeat_sec: Optional[float] = None, # lặp lại mỗi bao lâu (mặc định = threshold)
        safe_recovery_sec: float = 1.5,     # anti-flicker khi mũi/miệng hết che
    ):
        self.threshold_sec     = threshold_sec
        self.no_person_sec     = no_person_sec
        self.repeat_sec        = repeat_sec if repeat_sec is not None else threshold_sec
        self.safe_recovery_sec = safe_recovery_sec

        # --- trạng thái CHE ---
        self.covered_start: Optional[float]   = None
        self.covered_last_alert: Optional[float] = None
        self.covered_safe_since: Optional[float] = None
        # --- trạng thái MẤT NGƯỜI ---
        self.absent_start: Optional[float]    = None
        self.absent_last_alert: Optional[float] = None

    # --- backward-compat cho main.py (UI countdown) ---
    @property
    def occlusion_start(self) -> Optional[float]:
        """Mốc bắt đầu tình huống đang diễn ra (che hoặc mất người)."""
        return self.covered_start if self.covered_start is not None else self.absent_start

    @property
    def last_alert_at(self) -> Optional[float]:
        if self.covered_start is not None:
            return self.covered_last_alert
        return self.absent_last_alert

    # --- helpers ---
    def _reset_covered(self) -> None:
        self.covered_start = None
        self.covered_last_alert = None
        self.covered_safe_since = None

    def _reset_absent(self) -> None:
        self.absent_start = None
        self.absent_last_alert = None

    @staticmethod
    def _maybe_fire(now, start, last_alert, threshold, repeat):
        """Trả về (elapsed, should_alert, new_last_alert)."""
        elapsed = now - start
        if elapsed < threshold:
            return elapsed, False, last_alert
        if last_alert is None:
            return elapsed, True, now
        if now - last_alert >= repeat:
            return elapsed, True, now
        return elapsed, False, last_alert

    # --- main entrypoint ---
    def step(
        self,
        now: float,
        person_present: bool,
        covered: bool,
    ) -> StepResult:

        # === Không thấy người ===
        # KHÔNG còn gửi cảnh báo "mất người". Vẫn giữ STATE_NO_PERSON + đếm
        # elapsed để log/UI hiển thị "không thấy ai", nhưng should_alert luôn
        # False (không Telegram, không lưu ảnh cho tình huống này).
        if not person_present:
            self._reset_covered()
            if self.absent_start is None:
                self.absent_start = now
            elapsed = now - self.absent_start
            return StepResult(STATE_NO_PERSON, elapsed, False, TRIGGER_NO_PERSON)

        # === Có người ===
        self._reset_absent()

        if covered:
            self.covered_safe_since = None
            if self.covered_start is None:
                self.covered_start = now
            elapsed, should, self.covered_last_alert = self._maybe_fire(
                now, self.covered_start, self.covered_last_alert,
                self.threshold_sec, self.repeat_sec,
            )
            return StepResult(STATE_COVERED, elapsed, should, TRIGGER_COVERED)

        # Có người + không bị che
        if self.covered_start is None:
            self.covered_safe_since = None
            return StepResult(STATE_SAFE, 0.0, False, "")

        # Đang trong sự kiện che → anti-flicker: cần sạch liên tục đủ lâu mới reset.
        if self.covered_safe_since is None:
            self.covered_safe_since = now
        if now - self.covered_safe_since >= self.safe_recovery_sec:
            self._reset_covered()
            return StepResult(STATE_SAFE, 0.0, False, "")
        # Chưa đủ sạch → giữ COVERED, vẫn có thể re-alert.
        elapsed, should, self.covered_last_alert = self._maybe_fire(
            now, self.covered_start, self.covered_last_alert,
            self.threshold_sec, self.repeat_sec,
        )
        return StepResult(STATE_COVERED, elapsed, should, TRIGGER_COVERED)
