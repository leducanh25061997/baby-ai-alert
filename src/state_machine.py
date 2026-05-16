"""Pure state machine cho phát hiện ngạt thở.

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
    """Quản lý vòng đời cảnh báo ngạt thở.

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
    ):
        self.threshold_sec   = threshold_sec
        self.grace_sec       = grace_sec
        self.gone_reset_sec  = gone_reset_sec

        self.last_face_seen: Optional[float] = None
        self.occlusion_start: Optional[float] = None
        self.alert_sent      = False
        self.trigger_reason  = ""

    # --- helpers ---
    def _reset(self) -> None:
        self.occlusion_start = None
        self.alert_sent      = False
        self.trigger_reason  = ""

    def _maybe_alert(self, now: float) -> tuple[float, bool]:
        if self.occlusion_start is None:
            return 0.0, False
        elapsed = now - self.occlusion_start
        if elapsed >= self.threshold_sec and not self.alert_sent:
            self.alert_sent = True
            return elapsed, True
        return elapsed, False

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
                if self.occlusion_start is None:
                    self.occlusion_start = now
                    self.trigger_reason  = TRIGGER_HISTOGRAM
                elapsed, should = self._maybe_alert(now)
                return StepResult(STATE_ALERT, elapsed, should,
                                  self.trigger_reason or TRIGGER_HISTOGRAM)

            # Mặt OK, không bị che → reset alert state, tiếp tục giám sát
            self._reset()
            return StepResult(STATE_SAFE, 0.0, False, "")

        # === Nhánh 2: KHÔNG thấy mặt ===
        # Lưu ý đây chính là kịch bản nguy hiểm nhất (bị phủ kín).
        #
        # SAFETY-FIRST ORDERING: kiểm tra grace + occlusion_start TRƯỚC khi cho
        # YOLO=False override reset. Lý do thực tế trên Orange Pi:
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

        # 2b. Đã nghi rồi (occlusion_start set) + chưa quá gone_reset_sec →
        # tiếp tục đếm BẤT KỂ YOLO. Một khi đã vào trạng thái nghi, không
        # cho YOLO=False thắng — phải đợi face quay lại hoặc đủ ngưỡng alert.
        if time_since <= self.gone_reset_sec and self.occlusion_start is not None:
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
