"""Bộ phát hiện CHE mũi/miệng — đơn giản, robust, ít báo nhầm.

Triết lý sau khi viết lại: chỉ dùng 2 tín hiệu BỀN với mặt sạch:

  1. **Skin ratio** (tỉ lệ điểm ảnh màu da trong vùng mũi/miệng)
     - Vật che thật (chăn/gối/khăn/giấy/đồ chơi) → KHÔNG phải màu da → skin tụt mạnh.
     - Mặt sạch → vùng mũi/miệng vẫn là da → skin cao. Đây là tín hiệu CHÍNH.

  2. **Histogram correlation** (phân bố màu H,S so với baseline)
     - Bù cho vật che TÌNH CỜ có màu gần da: phân bố màu vẫn đổi mạnh.
     - Chỉ tính là "che" khi corr tụt RẤT sâu (< ngưỡng bảo thủ ~0.55) → mặt
       sạch dao động nhẹ (corr 0.85–0.99) KHÔNG bao giờ kích.

ĐÃ BỎ HẲN edge density + Laplacian variance. Hai tín hiệu đó đo *độ nét ảnh*,
không đo *có vật che* — chúng tụt về 0 trên mặt sạch/áp sát/hơi mờ và là nguyên
nhân chính gây báo nhầm "bị che" trên mặt nhìn rõ (xem events/ cũ: skin=1.0,
hist=0.96 mà vẫn bị vote che chỉ vì edge=0).

Quyết định "che" CHỈ dựa vào skin-drop (tín hiệu bền nhất với mặt sạch). Histogram
được tính để hiển thị/log nhưng KHÔNG dùng để quyết định — vùng miệng đổi histogram
mạnh khi bé cử động môi nên dễ gây báo nhầm.
Tính bền theo thời gian do main.py lo (SmoothingBuffer + ngưỡng thời gian).
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# MediaPipe FaceMesh landmark indices
NOSE_TIP     = 4
MOUTH_CENTER = 14

# Multi-landmark sampling cho mouth — robust khi mediapipe nhảy landmark.
MOUTH_LANDMARK_INDICES = [13, 14, 17]   # upper lip / mouth center / lower lip
NOSE_LANDMARK_INDICES  = [4]

PATCH_SIZE = 70

# Skin tone HSV ranges. Hue wraps quanh 0/180 (red), cần 2 range.
SKIN_HSV_RANGES = [
    (np.array([0,   30, 60],  dtype=np.uint8),
     np.array([25,  180, 255], dtype=np.uint8)),
    (np.array([160, 30, 60],  dtype=np.uint8),
     np.array([180, 180, 255], dtype=np.uint8)),
]

# Calibration:
MIN_CALIB_SAMPLES = 30
# Adaptive hist threshold = mean_self_corr - K_HIST*stddev, nhưng KẸP bảo thủ
# trong [0.30, 0.55] → chỉ kích khi corr tụt RẤT sâu (vật che thật), mặt sạch
# dao động nhẹ không bao giờ kích.
K_HIST          = 3.0
HIST_FLOOR_MIN  = 0.30
HIST_FLOOR_MAX  = 0.55
# Skin: che khi skin tụt > 50% so với baseline; sàn tuyệt đối 0.15.
SKIN_DROP_FRAC  = 0.50
SKIN_ABS_FLOOR  = 0.15

# Conservative baseline update:
BASELINE_UPDATE_CORR_MIN            = 0.92
BASELINE_UPDATE_RATE                = 0.005
FRAMES_AFTER_ALERT_TO_RESUME_UPDATE = 300   # ~10s ở 30fps


# ---------- helpers ----------

def extract_patch(frame, landmark, w: int, h: int, size: int = PATCH_SIZE):
    x = int(np.clip(landmark.x * w, 0, w - 1))
    y = int(np.clip(landmark.y * h, 0, h - 1))
    half = size // 2
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (size, size))


def compute_signals(patch_bgr) -> dict:
    """Trả về {hist, skin_ratio} cho 1 patch (chỉ 2 tín hiệu bền)."""
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)

    # HSV 2D histogram (H, S) — bỏ V để bớt nhạy với độ sáng
    hist = cv2.calcHist([hsv], [0, 1], None, [36, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    # Skin ratio
    masks = [cv2.inRange(hsv, lo, hi) for lo, hi in SKIN_HSV_RANGES]
    skin_mask = masks[0]
    for m in masks[1:]:
        skin_mask = cv2.bitwise_or(skin_mask, m)
    skin_ratio = float(np.count_nonzero(skin_mask)) / skin_mask.size

    return {'hist': hist.astype(np.float32), 'skin_ratio': skin_ratio}


def _hist_corr(h1, h2) -> float:
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def _aggregate(sigs_list: list[dict]) -> dict:
    """Gộp nhiều patch (mouth multi-landmark) bằng MEAN — mượt, ổn định."""
    hist = np.mean([s['hist'] for s in sigs_list], axis=0).astype(np.float32)
    return {
        'hist':       hist,
        'skin_ratio': float(np.mean([s['skin_ratio'] for s in sigs_list])),
    }


# ---------- data classes ----------

@dataclass
class PatchBaseline:
    """Baseline + ngưỡng cho 1 patch (mũi hoặc miệng)."""
    hist: np.ndarray
    skin_ratio: float
    hist_threshold: float    # corr dưới mức này → vote bị che
    skin_min_ratio: float    # skin_ratio dưới mức này → vote bị che


@dataclass
class CheckResult:
    """Kết quả 1 frame check."""
    occluded: bool
    nose_hist_corr: float
    mouth_hist_corr: float
    nose_skin_ratio: float
    mouth_skin_ratio: float
    nose_votes_for_occluded: int    # 0..2
    mouth_votes_for_occluded: int   # 0..2

    @property
    def total_votes(self) -> int:
        return self.nose_votes_for_occluded + self.mouth_votes_for_occluded


# ---------- main detector ----------

class OcclusionDetector:
    """Phát hiện che mũi/miệng dựa trên skin-ratio + histogram (bảo thủ)."""

    def __init__(self):
        self.nose: Optional[PatchBaseline]  = None
        self.mouth: Optional[PatchBaseline] = None
        self.is_ready                       = False
        self.calibration_quality            = 0.0
        self.calibration_message            = ""
        self._samples_nose: list[dict]      = []
        self._samples_mouth: list[dict]     = []
        self._frames_since_last_alert       = 10**9

    def reset(self):
        self.nose = None
        self.mouth = None
        self.is_ready = False
        self.calibration_quality = 0.0
        self.calibration_message = ""
        self._samples_nose.clear()
        self._samples_mouth.clear()
        self._frames_since_last_alert = 10**9

    def add_calibration_sample(self, frame, landmarks, w, h):
        if self.is_ready:
            return
        for idx in NOSE_LANDMARK_INDICES:
            patch = extract_patch(frame, landmarks[idx], w, h)
            if patch is not None:
                self._samples_nose.append(compute_signals(patch))
        mouth_patch_sigs = []
        for idx in MOUTH_LANDMARK_INDICES:
            patch = extract_patch(frame, landmarks[idx], w, h)
            if patch is not None:
                mouth_patch_sigs.append(compute_signals(patch))
        if mouth_patch_sigs:
            self._samples_mouth.append(_aggregate(mouth_patch_sigs))

    def finalize_calibration(self) -> tuple[bool, str]:
        if (len(self._samples_nose) < MIN_CALIB_SAMPLES
                or len(self._samples_mouth) < MIN_CALIB_SAMPLES):
            msg = (f"Quá ít sample (mũi={len(self._samples_nose)}, "
                   f"miệng={len(self._samples_mouth)}, cần >={MIN_CALIB_SAMPLES})")
            return False, msg

        nose_bl, nose_q, nose_msg = self._build_baseline(self._samples_nose, "mũi")
        if nose_bl is None:
            return False, nose_msg
        mouth_bl, mouth_q, mouth_msg = self._build_baseline(self._samples_mouth, "miệng")
        if mouth_bl is None:
            return False, mouth_msg

        self.nose = nose_bl
        self.mouth = mouth_bl
        self.calibration_quality = min(nose_q, mouth_q)
        self.calibration_message = (
            f"q={self.calibration_quality:.2f} | "
            f"hist_thresh nose={nose_bl.hist_threshold:.2f} mouth={mouth_bl.hist_threshold:.2f} | "
            f"skin_min nose={nose_bl.skin_min_ratio:.2f} mouth={mouth_bl.skin_min_ratio:.2f}"
        )
        self.is_ready = True
        return True, self.calibration_message

    def _build_baseline(self, samples: list[dict], label: str):
        hists = np.array([s['hist'] for s in samples])
        mean_hist = hists.mean(axis=0).astype(np.float32)
        cv2.normalize(mean_hist, mean_hist, 0, 1, cv2.NORM_MINMAX)

        self_corrs = np.array(
            [_hist_corr(h.astype(np.float32), mean_hist) for h in hists]
        )
        mean_corr = float(self_corrs.mean())
        std_corr  = float(self_corrs.std())
        # Kẹp bảo thủ: chỉ kích khi corr tụt rất sâu.
        hist_threshold = min(HIST_FLOOR_MAX,
                             max(HIST_FLOOR_MIN, mean_corr - K_HIST * std_corr))

        skin_ratios = np.array([s['skin_ratio'] for s in samples])
        mean_skin = float(skin_ratios.mean())
        std_skin  = float(skin_ratios.std())

        # Critical fail: skin quá thấp → landmark trỏ vào tóc/áo, không phải da.
        if mean_skin < 0.15:
            return None, 0.0, (
                f"❌ Calibration {label} kém: skin chỉ {mean_skin*100:.0f}% — "
                f"landmark có thể trỏ vào tóc/quần áo, không phải da. "
                f"Đảm bảo mặt trẻ nằm rõ trong khung, đủ sáng."
            )
        # Critical fail: histogram quá hỗn loạn.
        if std_corr > 0.25:
            return None, 0.0, (
                f"❌ Calibration {label} không ổn định: corr stddev={std_corr:.2f} "
                f"(quá cao). Giữ yên mặt trẻ + tránh đèn nhấp nháy + calibrate lại."
            )

        skin_min = max(SKIN_ABS_FLOOR, mean_skin * (1.0 - SKIN_DROP_FRAC))

        hist_q  = max(0.0, min(1.0, 1.0 - std_corr / 0.15))
        skin_q  = max(0.0, min(1.0, 1.0 - std_skin / 0.10))
        quality = (hist_q + skin_q) / 2

        baseline = PatchBaseline(
            hist=mean_hist,
            skin_ratio=mean_skin,
            hist_threshold=hist_threshold,
            skin_min_ratio=skin_min,
        )
        return baseline, quality, "OK"

    def check(self, frame, landmarks, w, h,
              prev_in_alert: bool) -> Optional[CheckResult]:
        """Check 1 frame. None nếu chưa ready hoặc landmark fail."""
        if not self.is_ready:
            return None

        nose_patches_sigs = []
        for idx in NOSE_LANDMARK_INDICES:
            patch = extract_patch(frame, landmarks[idx], w, h)
            if patch is not None:
                nose_patches_sigs.append(compute_signals(patch))
        if not nose_patches_sigs:
            return None
        nose_sigs = _aggregate(nose_patches_sigs)

        mouth_patches_sigs = []
        for idx in MOUTH_LANDMARK_INDICES:
            patch = extract_patch(frame, landmarks[idx], w, h)
            if patch is not None:
                mouth_patches_sigs.append(compute_signals(patch))
        if not mouth_patches_sigs:
            return None
        mouth_sigs = _aggregate(mouth_patches_sigs)

        nose_corr  = _hist_corr(nose_sigs['hist'],  self.nose.hist)
        mouth_corr = _hist_corr(mouth_sigs['hist'], self.mouth.hist)

        # SKIN-DROP là tín hiệu QUYẾT ĐỊNH: mặt sạch luôn có skin cao (bất kể
        # sáng/mờ/áp sát), vật che (chăn/gối/khăn/giấy) → skin tụt. Cả 2 ca báo
        # nhầm cũ (events/162626,162643) đều có skin=1.0 → skin-drop = False →
        # không báo. histogram chỉ để HIỂN THỊ/log (đặc biệt vùng miệng đổi mạnh
        # khi bé cử động môi → corr tụt oan, KHÔNG được dùng để quyết định).
        nose_skin_v = nose_sigs['skin_ratio']  < self.nose.skin_min_ratio
        nose_hist_v = nose_corr                < self.nose.hist_threshold
        nose_votes  = int(nose_skin_v) + int(nose_hist_v)

        mouth_skin_v = mouth_sigs['skin_ratio'] < self.mouth.skin_min_ratio
        mouth_hist_v = mouth_corr               < self.mouth.hist_threshold
        mouth_votes  = int(mouth_skin_v) + int(mouth_hist_v)

        # Che khi CẢ HAI vùng (mũi VÀ miệng) cùng mất màu da. Yêu cầu đồng thời
        # → loại báo nhầm khi chỉ 1 vùng tụt skin oan (ví dụ mặt người lớn có
        # râu/môi sẫm làm vùng miệng tụt skin dù mặt nhìn rõ). Vật che thật
        # (chăn/gối/khăn) thường phủ cả mũi lẫn miệng → cả hai cùng tụt.
        occluded = nose_skin_v and mouth_skin_v

        if occluded or prev_in_alert:
            self._frames_since_last_alert = 0
        else:
            self._frames_since_last_alert += 1

        can_update = (
            not occluded
            and not prev_in_alert
            and self._frames_since_last_alert > FRAMES_AFTER_ALERT_TO_RESUME_UPDATE
            and nose_corr  > BASELINE_UPDATE_CORR_MIN
            and mouth_corr > BASELINE_UPDATE_CORR_MIN
        )
        if can_update:
            self._update_baseline(self.nose,  nose_sigs)
            self._update_baseline(self.mouth, mouth_sigs)

        return CheckResult(
            occluded=occluded,
            nose_hist_corr=nose_corr,
            mouth_hist_corr=mouth_corr,
            nose_skin_ratio=nose_sigs['skin_ratio'],
            mouth_skin_ratio=mouth_sigs['skin_ratio'],
            nose_votes_for_occluded=nose_votes,
            mouth_votes_for_occluded=mouth_votes,
        )

    def _update_baseline(self, baseline: PatchBaseline, sigs: dict):
        r = BASELINE_UPDATE_RATE
        baseline.hist = ((1 - r) * baseline.hist
                         + r * sigs['hist']).astype(np.float32)
        baseline.skin_ratio = (1 - r) * baseline.skin_ratio + r * sigs['skin_ratio']
