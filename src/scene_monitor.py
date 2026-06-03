"""Phân tích frame ở mức TOÀN CỤC (rẻ CPU) — bổ trợ cho occlusion detector.

Dùng chung 1 ảnh xám downscale 0.25x cho mọi phép đo, làm 3 việc:

  1. BLUR GATE — variance-of-Laplacian CẢ KHUNG. Khi cả khung đột ngột mờ
     (autofocus hunting / motion blur ở FPS thấp) → báo "texture không đáng tin"
     để occlusion detector BỎ phiếu edge/lap vòng đó → diệt false-positive do mờ
     (đúng lớp lỗi đã gặp trên Pi 4: edge≈0, lap≈0 dù da rõ).
     Lưu ý quan trọng: vật/tay che THẬT chỉ làm mờ VÙNG MẶT, nền vẫn nét → độ
     nét toàn cục KHÔNG sụt → gate không kích → vẫn phát hiện che bình thường.
     Chỉ khi CẢ khung mờ (focus/motion) gate mới bật. → phân biệt tự nhiên.

  2. MOTION — mean abs-diff giữa 2 frame liên tiếp. Phục vụ phát hiện camera
     ĐƠ (frozen): diff ~ 0 tuyệt đối (cảnh sống luôn có nhiễu cảm biến > eps).

  3. LUMA — độ sáng trung bình, cho cảnh báo "quá tối" (watchdog).

Chỉ phụ thuộc cv2 + numpy → unit-test được độc lập (không cần mediapipe/telegram).
"""
from __future__ import annotations
import cv2


class SceneMonitor:
    def __init__(self, downscale: float = 0.25, blur_drop_frac: float = 0.45,
                 frozen_eps: float = 0.05, sharp_update_rate: float = 0.05):
        self.downscale         = downscale
        self.blur_drop_frac    = blur_drop_frac     # current < frac*baseline → mờ
        self.frozen_eps        = frozen_eps         # diff < eps → coi như đơ
        self.sharp_update_rate = sharp_update_rate

        self._prev_gray   = None
        self._sharp_ema   = None

        # đo gần nhất (cho UI/log)
        self.sharpness = 0.0
        self.motion    = None     # None = chưa có frame trước để so
        self.luma      = 0.0

    # ---------- đo mỗi frame ----------
    def analyze(self, frame):
        """Tính sharpness/motion/luma từ ảnh xám downscale. Gọi 1 lần/frame."""
        small = cv2.resize(frame, (0, 0), fx=self.downscale, fy=self.downscale,
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.luma      = float(gray.mean())
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            self.motion = float(cv2.absdiff(gray, self._prev_gray).mean())
        else:
            self.motion = None
        self._prev_gray = gray
        return self.sharpness, self.motion, self.luma

    # ---------- blur baseline (gọi khi frame CLEAR: calibrating / SAFE) ----------
    def update_sharp_baseline(self):
        if self._sharp_ema is None:
            self._sharp_ema = self.sharpness
        else:
            r = self.sharp_update_rate
            self._sharp_ema = (1 - r) * self._sharp_ema + r * self.sharpness

    @property
    def sharp_baseline(self):
        return self._sharp_ema

    @property
    def is_blurry(self) -> bool:
        if self._sharp_ema is None or self._sharp_ema <= 0:
            return False
        return self.sharpness < self.blur_drop_frac * self._sharp_ema

    @property
    def is_frozen_frame(self) -> bool:
        return self.motion is not None and self.motion < self.frozen_eps

    def reset(self):
        self._prev_gray = None
        self._sharp_ema = None
