"""FSM THUẦN cho thông báo FACE-PRESENCE (phát hiện có khuôn mặt ổn định).

TÁCH HẲN khỏi state machine AN TOÀN (state_machine.py: SAFE/COVERED/FACE_LOST/
NO_PERSON). Module này CHỈ lo một việc: khi trong khung xuất hiện ≥1 khuôn mặt
HỢP LỆ và ỔN ĐỊNH thì phát ĐÚNG MỘT event (text-only), và chỉ phát lần sau khi
TẤT CẢ mặt đã rời khung ổn định đủ lâu + cooldown đã hết.

ĐÂY CHỈ LÀ FACE DETECTION:
  - Không face recognition, không xác minh danh tính, không embedding.
  - Không lưu dữ liệu sinh trắc học, không khẳng định mặt là một người cụ thể.
  - Không tuyên bố liveness / anti-spoof (chỉ lọc nhẹ: confidence, kích thước
    bbox, tính hợp lệ hình học, ổn định theo thời gian).

Module THUẦN stdlib (không cv2/mediapipe/telegram) → unit-test độc lập. Mọi mốc
thời gian do CALLER truyền vào (`now`), caller dùng time.monotonic() nên FSM
KHÔNG bị ảnh hưởng khi đồng hồ hệ thống nhảy.

States:
  INITIALIZING          : camera/detector chưa sẵn sàng (chưa step nào ready).
  NO_FACE               : đang chạy, chưa có khuôn mặt hợp lệ.
  FACE_CANDIDATE        : có ≥1 detection đạt điều kiện nhưng chưa đủ ổn định.
  FACE_CONFIRMED        : đã đủ frame + thời gian xác nhận (chờ cooldown nếu cần).
  NOTIFIED              : vừa tạo ĐÚNG một event ở step này.
  WAITING_FOR_FACE_EXIT : đã thông báo; KHÔNG phát lại khi vẫn còn ≥1 mặt.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

# ---- States ----
STATE_INITIALIZING          = "INITIALIZING"
STATE_NO_FACE               = "NO_FACE"
STATE_FACE_CANDIDATE        = "FACE_CANDIDATE"
STATE_FACE_CONFIRMED        = "FACE_CONFIRMED"
STATE_NOTIFIED              = "NOTIFIED"
STATE_WAITING_FOR_FACE_EXIT = "WAITING_FOR_FACE_EXIT"


@dataclass
class FacePresenceEvent:
    """Event face-presence với ID DUY NHẤT. Caller dùng cùng ID này để retry
    Telegram — FSM KHÔNG tạo event mới trên mỗi frame."""
    id: str
    at: float       # monotonic timestamp lúc phát


@dataclass
class FacePresenceConfig:
    """Ngưỡng xác nhận/exit/cooldown cho face-presence. Tách RIÊNG, KHÔNG dùng
    chung CONFIRM_FRAMES của occlusion (ý nghĩa khác nhau)."""
    min_confidence: float = 0.6      # ngưỡng confidence detection
    min_area_ratio: float = 0.02     # diện tích bbox tối thiểu so với frame (chuẩn hóa)
    confirm_frames: int   = 4        # số lần scan THẤY mặt tối thiểu để xác nhận
    confirm_sec:    float = 1.0      # thời gian tối thiểu để xác nhận
    exit_frames:    int   = 5        # số lần scan KHÔNG mặt liên tiếp để coi là đã rời
    exit_sec:       float = 2.0      # thời gian không-mặt tối thiểu để coi là đã rời
    cooldown_sec:   float = 30.0     # cooldown giữa 2 event face-presence


# ---------- Lọc detection (thuần hình học, test được) ----------

def is_valid_detection(bbox: Tuple[float, float, float, float],
                       score: float,
                       cfg: FacePresenceConfig,
                       tol: float = 0.05) -> bool:
    """Một detection hợp lệ khi:
      - confidence ≥ ngưỡng;
      - bbox (xmin,ymin,w,h) CHUẨN HÓA 0..1, w/h dương;
      - nằm trong frame (cho phép tràn nhẹ `tol` vì mediapipe đôi khi vượt biên);
      - diện tích ≥ min_area_ratio (không quá nhỏ).
    bbox âm/tràn biên bất thường hoặc quá nhỏ → loại."""
    xmin, ymin, w, h = bbox
    if score < cfg.min_confidence:
        return False
    if w <= 0.0 or h <= 0.0:
        return False
    if xmin < -tol or ymin < -tol:
        return False
    if xmin + w > 1.0 + tol or ymin + h > 1.0 + tol:
        return False
    if w * h < cfg.min_area_ratio:
        return False
    return True


def count_valid_faces(detections: Optional[Sequence[Tuple[Tuple[float, float, float, float], float]]],
                      cfg: FacePresenceConfig,
                      tol: float = 0.05) -> int:
    """Đếm số mặt HỢP LỆ. detections=None (detector lỗi) → coi như 0 (caller nên
    truyền ready=False để FREEZE FSM thay vì tính là không-mặt)."""
    if not detections:
        return 0
    return sum(1 for (bbox, score) in detections
               if is_valid_detection(bbox, score, cfg, tol))


# ---------- FSM thuần ----------

class FacePresenceFSM:
    """FSM phát event face-presence. Chỉ MUTATE từ MỘT execution context (caller
    gọi step() tuần tự trong main loop) — không tự sinh thread."""

    def __init__(self, cfg: Optional[FacePresenceConfig] = None):
        self.cfg = cfg or FacePresenceConfig()
        self.state = STATE_INITIALIZING
        self._event_seq = 0
        self._last_event_t: Optional[float] = None
        # Tiến trình "đang hiện diện" (cộng dồn, chịu được vài frame miss).
        self._present_frames = 0
        self._present_since: Optional[float] = None
        # Tiến trình "đang vắng" (liên tiếp; reset khi thấy mặt).
        self._absent_frames = 0
        self._absent_since: Optional[float] = None

    # --- helpers ---
    def _reset_present(self) -> None:
        self._present_frames = 0
        self._present_since = None

    def _reset_absent(self) -> None:
        self._absent_frames = 0
        self._absent_since = None

    def _emit(self, now: float) -> FacePresenceEvent:
        self._event_seq += 1
        self._last_event_t = now
        return FacePresenceEvent(id=f"face-presence-{self._event_seq}", at=now)

    def _confirmed(self, now: float) -> bool:
        return (self._present_frames >= self.cfg.confirm_frames
                and self._present_since is not None
                and now - self._present_since >= self.cfg.confirm_sec)

    def _exited(self, now: float) -> bool:
        return (self._absent_frames >= self.cfg.exit_frames
                and self._absent_since is not None
                and now - self._absent_since >= self.cfg.exit_sec)

    def _cooldown_ok(self, now: float) -> bool:
        return (self._last_event_t is None
                or now - self._last_event_t >= self.cfg.cooldown_sec)

    # --- main entrypoint ---
    def step(self, now: float, valid_face_count: int,
             ready: bool = True) -> Optional[FacePresenceEvent]:
        """Tiến FSM 1 lần phân tích. Trả event NẾU vừa phát (đúng 1 lần), ngược
        lại None.

        ready=False → camera/detector chưa sẵn (hoặc đang fault): FREEZE — giữ
        nguyên state, KHÔNG advance counter, KHÔNG phát. Nhờ vậy camera
        unplug/reconnect KHÔNG bị hiểu nhầm thành 'mặt đã rời' → không event giả.
        """
        if not ready:
            return None

        if self.state == STATE_INITIALIZING:
            self.state = STATE_NO_FACE
            self._reset_present()
            self._reset_absent()

        present = valid_face_count >= 1

        # Cập nhật accumulator. KHÔNG reset tiến trình hiện diện khi miss 1 frame
        # (chống rung detector trong lúc xác nhận); chỉ exit khi vắng đủ lâu.
        if present:
            self._present_frames += 1
            if self._present_since is None:
                self._present_since = now
            self._reset_absent()
        else:
            self._absent_frames += 1
            if self._absent_since is None:
                self._absent_since = now

        # --- NO_FACE ---
        if self.state == STATE_NO_FACE:
            if present:
                self.state = STATE_FACE_CANDIDATE
            else:
                self._reset_present()   # giữ sạch khi chưa có mặt
            return None

        # --- FACE_CANDIDATE ---
        if self.state == STATE_FACE_CANDIDATE:
            if self._exited(now):
                # Vắng đủ lâu → xuất hiện thoáng qua, không xác nhận → quay NO_FACE.
                self.state = STATE_NO_FACE
                self._reset_present()
                return None
            if self._confirmed(now):
                self.state = STATE_FACE_CONFIRMED
                # rơi xuống xử lý CONFIRMED ngay trong step này
            else:
                return None

        # --- FACE_CONFIRMED ---
        if self.state == STATE_FACE_CONFIRMED:
            # Mặt đã rời trước khi kịp thông báo (đang chờ cooldown) → re-arm.
            if self._exited(now):
                self.state = STATE_NO_FACE
                self._reset_present()
                return None
            # Chỉ phát khi cooldown đã hết (lần đầu: last_event_t=None → phát ngay).
            if self._cooldown_ok(now):
                ev = self._emit(now)
                self.state = STATE_NOTIFIED
                return ev
            return None

        # --- NOTIFIED → chuyển sang chờ exit ở cùng step (không phát thêm) ---
        if self.state == STATE_NOTIFIED:
            self.state = STATE_WAITING_FOR_FACE_EXIT

        # --- WAITING_FOR_FACE_EXIT ---
        if self.state == STATE_WAITING_FOR_FACE_EXIT:
            # Re-arm CHỈ khi TẤT CẢ mặt đã rời ổn định đủ lâu (còn ≥1 mặt → reset
            # absent ở trên → không bao giờ exit).
            if self._exited(now):
                self.state = STATE_NO_FACE
                self._reset_present()
            return None

        return None


# ---------- Tiện ích concurrency/lifecycle (thuần, test được) ----------

class SingleFlight:
    """Đảm bảo CHỈ một phiên phân tích chạy tại một thời điểm (latest-frame
    semantics, KHÔNG queue). acquire() thất bại → bỏ frame, không tích hàng đợi."""

    def __init__(self):
        self._busy = False

    def acquire(self) -> bool:
        if self._busy:
            return False
        self._busy = True
        return True

    def release(self) -> None:
        self._busy = False


class OnceGuard:
    """Cho một hành động (release camera / close model) chạy ĐÚNG MỘT lần; gọi
    lại an toàn (idempotent) → chống double-dispose lúc shutdown."""

    def __init__(self):
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def run(self, fn) -> bool:
        if self._done:
            return False
        self._done = True
        fn()
        return True
