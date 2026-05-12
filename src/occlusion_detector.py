"""Multi-signal occlusion detector — phát hiện che mũi/miệng bằng vote.

4 tín hiệu yếu, mỗi cái yếu ở chỗ khác → vote chéo cho robust.

  1. **Histogram correlation** (HSV color distribution)
     - Mạnh: phát hiện thay đổi tổng thể về màu
     - Yếu: tay che mặt → skin tone giống → corr cao → không phát hiện được

  2. **Skin pixel ratio** (HSV skin segmentation)
     - Mạnh: chăn/gối thường có skin% rất thấp
     - Yếu: bàn tay là skin → không phân biệt

  3. **Edge density** (Canny edges/pixel)
     - Mạnh: chăn phẳng → ít edge
     - Yếu: tay có khớp/nếp nhăn → edge tương đương → không phát hiện

  4. **Laplacian variance** (intensity variance → "độ chi tiết nhỏ")
     - Mạnh: phân biệt **mặt** (môi+lông+lỗ mũi → high var) vs **tay back**
       (uniform skin → low var). Đây là discriminator chính cho tay che.
     - Yếu: ánh sáng đột ngột đổi (nhưng baseline_update đã handle)

**Vote**: occluded khi ≥2/4 tín hiệu trên cùng 1 patch đồng ý bị che.
Phải vượt qua 2 trong 4 đường tấn công khác chiều → robust với cả chăn (color
yếu) lẫn tay (texture yếu).

Calibration tính threshold ADAPTIVE theo phân bố quan sát được (mỗi user +
môi trường có ngưỡng riêng) và báo quality để user biết có nên recalibrate.
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# MediaPipe FaceMesh landmark indices
NOSE_TIP     = 4
MOUTH_CENTER = 14

# Multi-landmark sampling cho mouth — robust khi mediapipe nhảy landmark
# giữa các vị trí khác nhau trên tay che mặt. Lấy 3 patch: lip trên, giữa,
# lip dưới. Detection aggregate bằng MIN (worst-case patch) cho edge/lap_var.
MOUTH_LANDMARK_INDICES = [13, 14, 17]   # upper lip top / mouth center / lower lip bottom
NOSE_LANDMARK_INDICES  = [4]            # nose tip is enough (nhỏ hơn)

PATCH_SIZE = 50

# Skin tone HSV ranges. Hue wraps quanh 0/180 (red), cần 2 range.
# Tone Á/Âu nằm hầu hết trong [H 0-25, S 30-180, V 60-255].
SKIN_HSV_RANGES = [
    (np.array([0,   30, 60],  dtype=np.uint8),
     np.array([25,  180, 255], dtype=np.uint8)),
    (np.array([160, 30, 60],  dtype=np.uint8),
     np.array([180, 180, 255], dtype=np.uint8)),
]

# Canny edge thresholds
CANNY_LOW, CANNY_HIGH = 50, 150

# Calibration:
MIN_CALIB_SAMPLES        = 30
# Adaptive hist threshold = mean_self_corr - K_HIST * stddev. K=3 = ~99% interval.
K_HIST                   = 3.0
# Skin/edge/lap_var: alert khi drop > FRAC × baseline
SKIN_DROP_FRAC           = 0.45
EDGE_DROP_FRAC           = 0.30   # tighter
LAPVAR_DROP_FRAC         = 0.50   # tighter — bị che = drop ≥ 50% variance

# ABSOLUTE FLOORS — bảo đảm hand detection ổn định bất kể baseline thấp.
# Nếu signal < floor → vote chắc chắn dù relative drop chưa đủ.
# Giá trị calibrate cho real-world: hand back có lap_var ~10-60 / edge ~0.5-2%
EDGE_ABSOLUTE_FLOOR      = 0.020   # 2% pixel có edge — hand thường 0.5-1.5%
LAPVAR_ABSOLUTE_FLOOR    = 50.0    # hand thường 10-50, mặt thường 200+

# Conservative baseline update:
BASELINE_UPDATE_CORR_MIN = 0.92    # chỉ update khi rất confident SAFE
BASELINE_UPDATE_RATE     = 0.005   # chậm hơn để đỡ drift (cũ: 0.01)
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
    """Trả về {hist, skin_ratio, edge_density, lap_var} cho 1 patch."""
    hsv  = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    # 1. HSV 2D histogram (H, S) — không lấy V để bớt nhạy với độ sáng
    hist = cv2.calcHist([hsv], [0, 1], None, [36, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    # 2. Skin ratio
    masks = [cv2.inRange(hsv, lo, hi) for lo, hi in SKIN_HSV_RANGES]
    skin_mask = masks[0]
    for m in masks[1:]:
        skin_mask = cv2.bitwise_or(skin_mask, m)
    skin_ratio = float(np.count_nonzero(skin_mask)) / skin_mask.size

    # 3. Edge density (Canny)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    # 4. Laplacian variance — discriminator cho hand-vs-face.
    # Mặt: môi/lông/lỗ mũi → high variance. Tay back: uniform → low.
    # Đây là discriminator MẠNH NHẤT cho case tay che (mà 3 signal trên fail).
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        'hist': hist.astype(np.float32),
        'skin_ratio': skin_ratio,
        'edge_density': edge_density,
        'lap_var': lap_var,
    }


def _hist_corr(h1, h2) -> float:
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def _aggregate_avg(sigs_list: list[dict]) -> dict:
    """Aggregate cho CALIBRATION — dùng AVG để capture face state tiêu biểu."""
    hist = np.mean([s['hist'] for s in sigs_list], axis=0).astype(np.float32)
    return {
        'hist':         hist,
        'skin_ratio':   float(np.mean([s['skin_ratio']   for s in sigs_list])),
        'edge_density': float(np.mean([s['edge_density'] for s in sigs_list])),
        'lap_var':      float(np.mean([s['lap_var']      for s in sigs_list])),
    }


def _aggregate_for_detection(sigs_list: list[dict]) -> dict:
    """Aggregate cho DETECTION — robust với landmark drift:
       - Hist + skin: dùng MEAN (color-based, smoother)
       - Edge + lap_var: dùng MIN (texture-based, worst-case = most suspicious)
    """
    hist = np.mean([s['hist'] for s in sigs_list], axis=0).astype(np.float32)
    return {
        'hist':         hist,
        'skin_ratio':   float(np.mean([s['skin_ratio']  for s in sigs_list])),
        'edge_density': float(min([s['edge_density'] for s in sigs_list])),
        'lap_var':      float(min([s['lap_var']      for s in sigs_list])),
    }


# ---------- data classes ----------

@dataclass
class PatchBaseline:
    """Baseline + adaptive threshold cho 1 patch (mũi hoặc miệng)."""
    hist: np.ndarray
    skin_ratio: float
    edge_density: float
    lap_var: float
    hist_threshold: float       # corr dưới mức này → vote bị che
    skin_min_ratio: float       # skin_ratio dưới mức này → vote bị che
    edge_min_density: float     # edge_density dưới mức này → vote bị che
    lap_var_min: float          # lap_var dưới mức này → vote bị che


@dataclass
class CheckResult:
    """Kết quả 1 frame check."""
    occluded: bool                  # quyết định cuối sau vote
    nose_hist_corr: float
    mouth_hist_corr: float
    nose_skin_ratio: float
    mouth_skin_ratio: float
    nose_edge_density: float
    mouth_edge_density: float
    nose_lap_var: float
    mouth_lap_var: float
    nose_votes_for_occluded: int    # 0..4
    mouth_votes_for_occluded: int   # 0..4

    @property
    def total_votes(self) -> int:
        return self.nose_votes_for_occluded + self.mouth_votes_for_occluded


# ---------- main detector ----------

class OcclusionDetector:
    """Multi-signal detector + calibration quality gating + conservative update."""

    def __init__(self):
        self.nose: Optional[PatchBaseline]  = None
        self.mouth: Optional[PatchBaseline] = None
        self.is_ready                       = False
        self.calibration_quality            = 0.0   # 0..1
        self.calibration_message            = ""
        self._samples_nose: list[dict]      = []
        self._samples_mouth: list[dict]     = []
        # Theo dõi thời gian từ alert cuối — để không update baseline khi
        # vừa trong trạng thái alert.
        self._frames_since_last_alert       = 10**9

    def reset(self):
        """Reset toàn bộ để re-calibrate."""
        self.nose = None
        self.mouth = None
        self.is_ready = False
        self.calibration_quality = 0.0
        self.calibration_message = ""
        self._samples_nose.clear()
        self._samples_mouth.clear()
        self._frames_since_last_alert = 10**9

    def add_calibration_sample(self, frame, landmarks, w, h):
        """Mỗi frame:
        - Nose: 1 landmark (NOSE_TIP) → 1 sample
        - Mouth: 3 landmark (upper/center/lower lip) → aggregate = AVG → 1 sample
        """
        if self.is_ready:
            return

        # Nose
        for idx in NOSE_LANDMARK_INDICES:
            patch = extract_patch(frame, landmarks[idx], w, h)
            if patch is not None:
                self._samples_nose.append(compute_signals(patch))

        # Mouth: aggregate multi-landmark with AVG (capture typical face state)
        mouth_patch_sigs = []
        for idx in MOUTH_LANDMARK_INDICES:
            patch = extract_patch(frame, landmarks[idx], w, h)
            if patch is not None:
                mouth_patch_sigs.append(compute_signals(patch))
        if mouth_patch_sigs:
            agg = _aggregate_avg(mouth_patch_sigs)
            self._samples_mouth.append(agg)

    def finalize_calibration(self) -> tuple[bool, str]:
        """Tính baseline + threshold + quality. Trả về (success, message)."""
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

        # Adaptive hist threshold dựa trên phân bố tự-tương-quan
        self_corrs = np.array(
            [_hist_corr(h.astype(np.float32), mean_hist) for h in hists]
        )
        mean_corr = float(self_corrs.mean())
        std_corr  = float(self_corrs.std())
        hist_threshold = max(0.30, mean_corr - K_HIST * std_corr)

        skin_ratios = np.array([s['skin_ratio'] for s in samples])
        edge_dens   = np.array([s['edge_density'] for s in samples])
        lap_vars    = np.array([s['lap_var']      for s in samples])
        mean_skin   = float(skin_ratios.mean())
        std_skin    = float(skin_ratios.std())
        mean_edge   = float(edge_dens.mean())
        mean_lap    = float(lap_vars.mean())

        # Critical fail: skin quá thấp → landmark trỏ vào tóc/áo, không phải da
        if mean_skin < 0.15:
            return None, 0.0, (
                f"❌ Calibration {label} kém: skin chỉ {mean_skin*100:.0f}% — "
                f"landmark có thể trỏ vào tóc/quần áo, không phải da. "
                f"Đảm bảo mặt trẻ nằm rõ trong khung, đủ sáng."
            )

        # Critical fail: histogram quá hỗn loạn
        if std_corr > 0.25:
            return None, 0.0, (
                f"❌ Calibration {label} không ổn định: corr stddev={std_corr:.2f} "
                f"(quá cao). Giữ yên mặt trẻ + tránh đèn nhấp nháy + calibrate lại."
            )

        skin_min    = max(0.05,                  mean_skin * (1.0 - SKIN_DROP_FRAC))
        # Absolute floor → đảm bảo hand luôn vote dù baseline thấp
        edge_min    = max(EDGE_ABSOLUTE_FLOOR,   mean_edge * (1.0 - EDGE_DROP_FRAC))
        lap_var_min = max(LAPVAR_ABSOLUTE_FLOOR, mean_lap  * (1.0 - LAPVAR_DROP_FRAC))

        # Quality 0..1 dựa trên độ ổn định
        hist_q = max(0.0, min(1.0, 1.0 - std_corr / 0.15))
        skin_q = max(0.0, min(1.0, 1.0 - std_skin / 0.10))
        quality = (hist_q + skin_q) / 2

        baseline = PatchBaseline(
            hist=mean_hist,
            skin_ratio=mean_skin,
            edge_density=mean_edge,
            lap_var=mean_lap,
            hist_threshold=hist_threshold,
            skin_min_ratio=skin_min,
            edge_min_density=edge_min,
            lap_var_min=lap_var_min,
        )
        return baseline, quality, "OK"

    def check(self, frame, landmarks, w, h,
              prev_in_alert: bool) -> Optional[CheckResult]:
        """Check 1 frame. Trả về None nếu chưa ready hoặc landmark fail.

        prev_in_alert: trạng thái ALERT của frame TRƯỚC. Dùng để biết có an
        toàn để update baseline không (không update khi vừa ra khỏi alert).
        """
        if not self.is_ready:
            return None

        # Nose: single landmark
        nose_patches_sigs = []
        for idx in NOSE_LANDMARK_INDICES:
            patch = extract_patch(frame, landmarks[idx], w, h)
            if patch is not None:
                nose_patches_sigs.append(compute_signals(patch))
        if not nose_patches_sigs:
            return None
        nose_sigs = _aggregate_for_detection(nose_patches_sigs)

        # Mouth: multi-landmark, MIN aggregation cho edge/lap_var → robust với
        # landmark drift trên bề mặt tay
        mouth_patches_sigs = []
        for idx in MOUTH_LANDMARK_INDICES:
            patch = extract_patch(frame, landmarks[idx], w, h)
            if patch is not None:
                mouth_patches_sigs.append(compute_signals(patch))
        if not mouth_patches_sigs:
            return None
        mouth_sigs = _aggregate_for_detection(mouth_patches_sigs)

        # === Compute correlations ===
        nose_corr  = _hist_corr(nose_sigs['hist'],  self.nose.hist)
        mouth_corr = _hist_corr(mouth_sigs['hist'], self.mouth.hist)

        # === Per-signal votes for "occluded" ===
        # Mỗi patch có 4 vote: hist / skin / edge / lap_var
        nose_hist_v = nose_corr                  < self.nose.hist_threshold
        nose_skin_v = nose_sigs['skin_ratio']    < self.nose.skin_min_ratio
        nose_edge_v = nose_sigs['edge_density']  < self.nose.edge_min_density
        nose_lap_v  = nose_sigs['lap_var']       < self.nose.lap_var_min
        nose_votes  = (int(nose_hist_v) + int(nose_skin_v)
                       + int(nose_edge_v) + int(nose_lap_v))

        mouth_hist_v = mouth_corr                 < self.mouth.hist_threshold
        mouth_skin_v = mouth_sigs['skin_ratio']   < self.mouth.skin_min_ratio
        mouth_edge_v = mouth_sigs['edge_density'] < self.mouth.edge_min_density
        mouth_lap_v  = mouth_sigs['lap_var']      < self.mouth.lap_var_min
        mouth_votes  = (int(mouth_hist_v) + int(mouth_skin_v)
                        + int(mouth_edge_v) + int(mouth_lap_v))

        # Final: occluded nếu BẤT KỲ patch nào có ≥2/4 vote
        occluded = (nose_votes >= 2) or (mouth_votes >= 2)

        # === Track frames since alert ===
        if occluded or prev_in_alert:
            self._frames_since_last_alert = 0
        else:
            self._frames_since_last_alert += 1

        # === Conservative baseline update ===
        # Chỉ update khi: KHÔNG occluded + đã đủ thời gian từ alert cuối + corr rất cao
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
            nose_edge_density=nose_sigs['edge_density'],
            mouth_edge_density=mouth_sigs['edge_density'],
            nose_lap_var=nose_sigs['lap_var'],
            mouth_lap_var=mouth_sigs['lap_var'],
            nose_votes_for_occluded=nose_votes,
            mouth_votes_for_occluded=mouth_votes,
        )

    def _update_baseline(self, baseline: PatchBaseline, sigs: dict):
        r = BASELINE_UPDATE_RATE
        baseline.hist = ((1 - r) * baseline.hist
                         + r * sigs['hist']).astype(np.float32)
        baseline.skin_ratio   = (1 - r) * baseline.skin_ratio   + r * sigs['skin_ratio']
        baseline.edge_density = (1 - r) * baseline.edge_density + r * sigs['edge_density']
        baseline.lap_var      = (1 - r) * baseline.lap_var      + r * sigs['lap_var']
