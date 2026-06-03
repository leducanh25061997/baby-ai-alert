"""Pure state machine cho phát hiện che mũi/miệng.

Module này KHÔNG có dependency nào ngoài stdlib — vì vậy có thể unit test
độc lập với cv2/mediapipe/telegram. main.py gọi step() mỗi frame và xử
lý kết quả (vẽ UI, gửi Telegram, lưu event).
"""
from dataclasses import dataclass
from typing import Optional


# Trạng thái cấp UI
STATE_NO_FACE     = "NO_FACE"
STATE_CALIBRATING = "CALIBRATING"
STATE_SAFE        = "SAFE"
STATE_ALERT       = "ALERT"

# Lý do trigger
TRIGGER_HISTOGRAM = "histogram"
TRIGGER_FACE_LOST = "face_lost"


@dataclass
class StepResult:
    state: str
    elapsed: float
    should_alert: bool
    trigger: str


class OcclusionStateMachine:
    """Quản lý vòng đời cảnh báo che mũi/miệng.

    Đầu vào mỗi frame:
      - now: timestamp (float seconds)
      - face_present: face_mesh có thấy mặt không
      - occluded_by_histogram: histogram mũi/miệng nói có bị che không
      - person_in_frame: YOLO có thấy người không (None = không có YOLO)

    Đầu ra: StepResult — caller đọc state để vẽ UI, đọc should_alert để
    gửi Telegram + lưu event.
    """

    def __init__(
        self,
        threshold_sec: float = 15.0,
        grace_sec: float = 1.5,
        gone_reset_sec: float = 25.0,
        repeat_alert_sec: Optional[float] = None,
        safe_recovery_sec: float = 1.5,
    ):
        self.threshold_sec   = threshold_sec
        self.grace_sec       = grace_sec
        self.gone_reset_sec  = gone_reset_sec
        # Bao lâu re-alert khi vẫn còn bị che. Mặc định = threshold_sec
        # (15s che → alert; tiếp tục che thêm 15s → alert lại; ...).
        self.repeat_alert_sec  = (
            repeat_alert_sec if repeat_alert_sec is not None else threshold_sec
        )
        # Anti-flicker: khi đang trong ALERT, cần thấy mặt sạch liên tục
        # ≥ safe_recovery_sec mới reset occlusion_start. Tránh MediaPipe
        # flicker thấy/mất mặt làm counter bị reset oan.
        self.safe_recovery_sec = safe_recovery_sec

        self.last_face_seen: Optional[float] = None
        self.occlusion_start: Optional[float] = None
        # Thời điểm alert gần nhất đã gửi (None = chưa gửi lần nào trong
        # occlusion event hiện tại). Dùng cho repeat-alert mỗi 15s.
        self.last_alert_at: Optional[float] = None
        # Thời điểm bắt đầu safe streak khi đang trong occlusion event
        # (None = chưa thấy frame safe nào sau khi nghi). Dùng anti-flicker.
        self.safe_since: Optional[float] = None
        self.trigger_reason  = ""

    # --- helpers ---
    def _reset(self) -> None:
        self.occlusion_start = None
        self.last_alert_at   = None
        self.safe_since      = None
        self.trigger_reason  = ""

    def _maybe_alert(self, now: float) -> tuple[float, bool]:
        """Trả về (elapsed, should_alert).

        Bắn alert lần đầu khi elapsed ≥ threshold_sec.
        Sau đó, mỗi repeat_alert_sec bắn lại 1 lần nếu vẫn còn trong occlusion.
        """
        if self.occlusion_start is None:
            return 0.0, False
        elapsed = now - self.occlusion_start
        if elapsed < self.threshold_sec:
            return elapsed, False
        # Đủ ngưỡng → alert lần đầu hoặc lần lặp
        if self.last_alert_at is None:
            self.last_alert_at = now
            return elapsed, True
        if now - self.last_alert_at >= self.repeat_alert_sec:
            self.last_alert_at = now
            return elapsed, True
        return elapsed, False

    @property
    def alert_sent(self) -> bool:
        """Backward-compat: True nếu đã bắn ít nhất 1 alert trong event này."""
        return self.last_alert_at is not None

    # --- main entrypoint ---
    def step(
        self,
        now: float,
        face_present: bool,
        occluded_by_histogram: bool,
        person_in_frame: Optional[bool] = None,
    ) -> StepResult:

        # === Nhánh 1: thấy mặt ===
        if face_present:
            self.last_face_seen = now

            if occluded_by_histogram:
                # Đang bị che → clear safe streak (nếu có)
                self.safe_since = None
                if self.occlusion_start is None:
                    self.occlusion_start = now
                    self.trigger_reason  = TRIGGER_HISTOGRAM
                elapsed, should = self._maybe_alert(now)
                return StepResult(STATE_ALERT, elapsed, should,
                                  self.trigger_reason or TRIGGER_HISTOGRAM)

            # face_present + not occluded
            # Nếu chưa từng nghi (occlusion_start is None) → SAFE bình thường.
            if self.occlusion_start is None:
                self.safe_since = None
                return StepResult(STATE_SAFE, 0.0, False, "")

            # Đang trong occlusion event → ANTI-FLICKER: cần thấy mặt sạch
            # liên tục ≥ safe_recovery_sec mới reset. Tránh MediaPipe flicker
            # 1 frame "thấy mặt sạch" làm counter bị xóa oan.
            if self.safe_since is None:
                self.safe_since = now
            if now - self.safe_since >= self.safe_recovery_sec:
                self._reset()
                return StepResult(STATE_SAFE, 0.0, False, "")
            # Chưa đủ safe → giữ ALERT, tiếp tục đếm + có thể re-alert
            elapsed, should = self._maybe_alert(now)
            return StepResult(STATE_ALERT, elapsed, should,
                              self.trigger_reason or TRIGGER_HISTOGRAM)

        # === Nhánh 2: KHÔNG thấy mặt ===
        # Không thấy mặt thì không thể tính "safe streak"
        self.safe_since = None

        # Lưu ý đây chính là kịch bản nguy hiểm nhất (bị phủ kín).
        #
        # SAFETY-FIRST ORDERING: kiểm tra grace + occlusion_start TRƯỚC khi cho
        # YOLO=False override reset. Lý do thực tế (camera nhìn top-down):
        #   - Camera đặt nhìn TOP-DOWN xuống mặt trẻ nằm ngửa.
        #   - YOLO COCO 'person' class chủ yếu train trên người đứng/ngồi →
        #     top-down view trẻ → YOLO HAY trả False (miss detect, không
        #     phải vì trẻ rời khung).
        #   - Khi giấy/chăn che kín mặt: MediaPipe mất face_present + YOLO
        #     trả False sai → nếu reset ngay sẽ MISS alert hoàn toàn.
        # Vì vậy: trong grace + đã nghi che → tin face_lost path, kệ YOLO.

        time_since = (
            now - self.last_face_seen
            if self.last_face_seen is not None else float("inf")
        )

        # 2a. Mặt vừa biến mất (trong grace) → khởi động đếm BẤT KỂ YOLO.
        # Grace quá ngắn (1.5s) để tin một frame YOLO=False.
        if time_since <= self.grace_sec:
            if self.occlusion_start is None:
                self.occlusion_start = now - time_since
                self.trigger_reason  = TRIGGER_FACE_LOST
            elapsed, should = self._maybe_alert(now)
            return StepResult(STATE_ALERT, elapsed, should,
                              self.trigger_reason or TRIGGER_FACE_LOST)

        # 2b. Đã nghi che (occlusion_start set) → tiếp tục đếm BẤT KỂ YOLO khi:
        #   (a) ĐÃ bắn ≥1 alert trong event này (last_alert_at set) →
        #       KHÔNG BAO GIỜ bỏ cuộc vì time-out. Mặt bị phủ kín làm MediaPipe
        #       mất tracking + YOLO top-down trả False sai là kịch bản che kín
        #       chết người: phải re-alert mỗi repeat_alert_sec cho tới khi mặt
        #       quay lại sạch (nhánh 1 mới được reset). Chấp nhận false-positive
        #       (bế trẻ đi sau khi đã alert mà không quay lại khung) để TUYỆT
        #       ĐỐI KHÔNG MISS cảnh báo thật. Đây là yêu cầu an toàn cốt lõi.
        #   (b) HOẶC chưa alert nhưng còn trong gone_reset_sec window.
        # Một khi đã vào trạng thái nghi, không cho YOLO=False thắng.
        if self.occlusion_start is not None and (
            self.last_alert_at is not None or time_since <= self.gone_reset_sec
        ):
            elapsed, should = self._maybe_alert(now)
            return StepResult(STATE_ALERT, elapsed, should,
                              self.trigger_reason or TRIGGER_FACE_LOST)

        # 2c. YOLO khẳng định CÓ người dù không thấy mặt → kiểu gì cũng nghi
        # bị phủ. Tiếp tục đếm dù grace hết.
        if person_in_frame is True:
            if self.occlusion_start is None:
                self.occlusion_start = now - min(time_since, self.gone_reset_sec)
                self.trigger_reason  = TRIGGER_FACE_LOST
            elapsed, should = self._maybe_alert(now)
            return StepResult(STATE_ALERT, elapsed, should,
                              self.trigger_reason or TRIGGER_FACE_LOST)

        # 2d. YOLO khẳng định KHÔNG có người, đã ngoài grace + chưa từng nghi
        # che → trẻ rời khung thật, reset.
        if person_in_frame is False:
            self._reset()
            return StepResult(STATE_NO_FACE, 0.0, False, "")

        # 2e. Mất mặt quá lâu, không có YOLO khẳng định → coi như rời khung.
        self._reset()
        return StepResult(STATE_NO_FACE, 0.0, False, "")
