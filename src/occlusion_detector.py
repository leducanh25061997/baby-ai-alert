"""Multi-signal occlusion detector — phát hiện che mũi/miệng bằng vote.

3 tín hiệu yếu, mỗi cái có điểm yếu khác nhau → vote chéo cho robust.

  1. **Histogram correlation** (HSV color distribution)
     - Mạnh: phát hiện thay đổi tổng thể về màu
     - Yếu: dễ nhầm khi tay che (skin tone giống mặt)

  2. **Skin pixel ratio** (HSV skin segmentation)
     - Mạnh: chăn/gối thường có skin% rất thấp
     - Yếu: bàn tay là skin → không phân biệt được

  3. **Edge density** (Canny edges/pixel)
     - Mạnh: mặt có nhiều chi tiết (môi, lỗ mũi); chăn/tay phẳng → edge thấp
     - Yếu: chăn họa tiết phức tạp có thể có nhiều edge

**Vote**: occluded khi ≥2 tín hiệu (trên cùng 1 patch) đồng ý bị che. Như vậy
phải 2/3 đường tấn công lệch khỏi baseline mới alert → giảm false positive đáng kể.

Calibration tính threshold ADAPTIVE theo phân bố trong giai đoạn calibrate (mỗi
user + môi trường → ngưỡng riêng) và báo quality để user biết có nên recalibrate.
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# MediaPipe FaceMesh landmark indices
NOSE_TIP     = 4
MOUTH_CENTER = 14

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
# Skin/edge: alert khi drop > FRAC × baseline
SKIN_DROP_FRAC           = 0.45
EDGE_DROP_FRAC           = 0.50

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
    """Trả về {hist, skin_ratio, edge_density} cho 1 patch."""
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)

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
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    return {
        'hist': hist.astype(np.float32),
        'skin_ratio': skin_ratio,
        'edge_density': edge_density,
    }


def _hist_corr(h1, h2) -> float:
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


# ---------- data classes ----------

@dataclass
class PatchBaseline:
    """Baseline + adaptive threshold cho 1 patch (mũi hoặc miệng)."""
    hist: np.ndarray
    skin_ratio: float
    edge_density: float
    hist_threshold: float       # corr dưới mức này → vote bị che
    skin_min_ratio: float       # skin_ratio dưới mức này → vote bị che
    edge_min_density: float     # edge_density dưới mức này → vote bị che


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
    nose_votes_for_occluded: int    # 0..3
    mouth_votes_for_occluded: int   # 0..3

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
        if self.is_ready:
            return
        for idx, samples in ((NOSE_TIP, self._samples_nose),
                             (MOUTH_CENTER, self._samples_mouth)):
            patch = extract_patch(frame, landmarks[idx], w, h)
            if patch is None:
                continue
            samples.append(compute_signals(patch))

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
        mean_skin   = float(skin_ratios.mean())
        std_skin    = float(skin_ratios.std())
        mean_edge   = float(edge_dens.mean())

        # Critical fail: skin quá thấp → landmark có thể trỏ vào tóc/áo, không phải da
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

        skin_min = max(0.05, mean_skin * (1.0 - SKIN_DROP_FRAC))
        edge_min = max(0.005, mean_edge * (1.0 - EDGE_DROP_FRAC))

        # Quality 0..1 dựa trên độ ổn định
        hist_q = max(0.0, min(1.0, 1.0 - std_corr / 0.15))
        skin_q = max(0.0, min(1.0, 1.0 - std_skin / 0.10))
        quality = (hist_q + skin_q) / 2

        baseline = PatchBaseline(
            hist=mean_hist,
            skin_ratio=mean_skin,
            edge_density=mean_edge,
            hist_threshold=hist_threshold,
            skin_min_ratio=skin_min,
            edge_min_density=edge_min,
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

        nose_patch  = extract_patch(frame, landmarks[NOSE_TIP],    w, h)
        mouth_patch = extract_patch(frame, landmarks[MOUTH_CENTER], w, h)
        if nose_patch is None or mouth_patch is None:
            return None

        nose_sigs  = compute_signals(nose_patch)
        mouth_sigs = compute_signals(mouth_patch)

        # === Compute correlations ===
        nose_corr  = _hist_corr(nose_sigs['hist'],  self.nose.hist)
        mouth_corr = _hist_corr(mouth_sigs['hist'], self.mouth.hist)

        # === Per-signal votes for "occluded" ===
        # Mỗi patch có 3 vote: hist / skin / edge
        nose_hist_v = nose_corr                  < self.nose.hist_threshold
        nose_skin_v = nose_sigs['skin_ratio']    < self.nose.skin_min_ratio
        nose_edge_v = nose_sigs['edge_density']  < self.nose.edge_min_density
        nose_votes  = int(nose_hist_v) + int(nose_skin_v) + int(nose_edge_v)

        mouth_hist_v = mouth_corr                 < self.mouth.hist_threshold
        mouth_skin_v = mouth_sigs['skin_ratio']   < self.mouth.skin_min_ratio
        mouth_edge_v = mouth_sigs['edge_density'] < self.mouth.edge_min_density
        mouth_votes  = int(mouth_hist_v) + int(mouth_skin_v) + int(mouth_edge_v)

        # Final: occluded nếu BẤT KỲ patch nào có ≥2/3 vote (majority trong patch)
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
            nose_votes_for_occluded=nose_votes,
            mouth_votes_for_occluded=mouth_votes,
        )

    def _update_baseline(self, baseline: PatchBaseline, sigs: dict):
        r = BASELINE_UPDATE_RATE
        baseline.hist = ((1 - r) * baseline.hist
                         + r * sigs['hist']).astype(np.float32)
        baseline.skin_ratio   = (1 - r) * baseline.skin_ratio   + r * sigs['skin_ratio']
        baseline.edge_density = (1 - r) * baseline.edge_density + r * sigs['edge_density']
