"""Phân tích frame ở mức TOÀN CỤC (rẻ CPU) — bổ trợ cho occlusion detector.

Dùng chung 1 ảnh xám downscale 0.25x cho mọi phép đo, làm 3 việc:

  1. BLUR GATE — variance-of-Laplacian CẢ KHUNG. Khi cả khung đột ngột mờ
     (autofocus hunting / motion blur ở FPS thấp) → báo "texture không đáng tin"
     để occlusion detector BỎ phiếu edge/lap vòng đó → diệt false-positive do mờ
     (đúng lớp lỗi đã gặp trên Pi 4: edge≈0, lap≈0 dù da rõ).
     Lưu ý quan trọng: vật/tay che THẬT chỉ làm mờ VÙNG MẶT, nền vẫn nét → độ
     nét toàn cục KHÔNG sụt → gate không kích → vẫn phát hiện che bình thường.
     Chỉ khi CẢ khung mờ (focus/motion) gate mới bật. → phân biệt tự nhiên.

  2. MOTION — mean abs-diff giữa 2 frame liên tiếp. Phục vụ:
       - "motion-absence" (nghi bất động / ngưng thở) khi vắng cử động quá lâu.
       - phát hiện camera ĐƠ (frozen): diff ~ 0 tuyệt đối (cảnh sống luôn có
         nhiễu cảm biến > eps).

  3. LUMA — độ sáng trung bình, cho cảnh báo "quá tối" (watchdog).

Chỉ phụ thuộc cv2 + numpy → unit-test được độc lập (không cần mediapipe/telegram).
"""
from __future__ import annotations
import cv2
import numpy as np


class SceneMonitor:
    def __init__(self, downscale: float = 0.25, blur_drop_frac: float = 0.45,
                 motion_floor: float = 0.4, motion_k: float = 4.0,
                 frozen_eps: float = 0.05, sharp_update_rate: float = 0.05):
        self.downscale         = downscale
        self.blur_drop_frac    = blur_drop_frac     # current < frac*baseline → mờ
        self.motion_floor      = motion_floor       # sàn tuyệt đối cho "có cử động"
        self.motion_k          = motion_k           # threshold = mean + k*std (calib)
        self.frozen_eps        = frozen_eps         # diff < eps → coi như đơ
        self.sharp_update_rate = sharp_update_rate

        self._prev_gray   = None
        self._sharp_ema   = None
        self._calib_motions: list[float] = []
        self.motion_threshold = None

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

    # ---------- motion threshold (calibrate cùng pha calibration) ----------
    def add_calib_motion(self):
        if self.motion is not None:
            self._calib_motions.append(self.motion)

    def finalize_motion_threshold(self) -> float:
        """Ngưỡng 'có cử động' = mean + k*std của motion lúc calibrate (nhiễu nền
        của camera) + sàn tuyệt đối. Cử động thật (kể cả thở nhẹ) vượt ngưỡng này;
        cảnh tĩnh/ngưng thở thì không → đo được 'vắng cử động'."""
        if self._calib_motions:
            arr = np.asarray(self._calib_motions, dtype=np.float64)
            self.motion_threshold = max(
                self.motion_floor,
                float(arr.mean() + self.motion_k * arr.std()),
            )
        else:
            self.motion_threshold = self.motion_floor
        return self.motion_threshold

    @property
    def has_motion(self) -> bool:
        # Chưa đo được / chưa calibrate → coi như CÓ cử động (an toàn: không bao
        # giờ báo "bất động" nhầm khi chưa đủ dữ liệu).
        if self.motion is None or self.motion_threshold is None:
            return True
        return self.motion > self.motion_threshold

    @property
    def is_frozen_frame(self) -> bool:
        return self.motion is not None and self.motion < self.frozen_eps

    def reset(self):
        self._prev_gray = None
        self._sharp_ema = None
        self._calib_motions.clear()
        self.motion_threshold = None
