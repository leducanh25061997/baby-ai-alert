"""Logic ĐO ĐẠC thuần — KHÔNG phụ thuộc cv2/mediapipe/sklearn/pandas.

Tách để unit-test độc lập (giống state_machine.py / alert_policy.py).

Dùng cho HƯỚNG A: đo precision/recall/false-alarm + ROC của bộ phát hiện che
trên dữ liệu thật đã gán nhãn, nhằm:
  (1) Biến "em nghĩ robust" → "em ĐO được X% recall ở Y% báo nhầm".
  (2) Chọn NGƯỠNG TỐI ƯU thay cho con số đặt-bằng-tay (vote≥2, các DROP_FRAC...).

Quy ước nhãn: 1 = BỊ CHE (positive), 0 = KHÔNG che (negative).
Dự đoán dương khi score >= threshold.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Confusion:
    tp: int   # che, đoán che
    fp: int   # không che, đoán che  → BÁO NHẦM (false alarm)
    tn: int   # không che, đoán không
    fn: int   # che, đoán không       → BỎ LỌT (miss — nguy hiểm nhất)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:        # = TPR = sensitivity (bắt được bao nhiêu % ca che)
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def fpr(self) -> float:           # false-alarm rate (báo nhầm trên ca KHÔNG che)
        d = self.fp + self.tn
        return self.fp / d if d else 0.0

    @property
    def specificity(self) -> float:
        d = self.fp + self.tn
        return self.tn / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / n if n else 0.0


def confusion_at(scores, labels, threshold) -> Confusion:
    """Đoán dương khi score >= threshold."""
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if   pred == 1 and y == 1: tp += 1
        elif pred == 1 and y == 0: fp += 1
        elif pred == 0 and y == 0: tn += 1
        else:                      fn += 1
    return Confusion(tp, fp, tn, fn)


@dataclass
class OperatingPoint:
    threshold: float
    confusion: Confusion

    @property
    def fpr(self) -> float: return self.confusion.fpr

    @property
    def tpr(self) -> float: return self.confusion.recall


def sweep(scores, labels, thresholds) -> list:
    """Tính điểm vận hành cho từng ngưỡng cho trước (vd [0,1,2,3,4] cho vote)."""
    return [OperatingPoint(float(t), confusion_at(scores, labels, t))
            for t in thresholds]


def roc_curve(scores, labels) -> list:
    """Đường ROC: quét mọi ngưỡng từ 'bắt không gì' (0,0) tới 'bắt tất' (1,1).

    Trả list OperatingPoint sắp theo fpr tăng dần, đã chèn 2 đầu mút (0,0)/(1,1)."""
    if not scores:
        return []
    uniq = sorted(set(scores))
    # Ngưỡng trên max → không đoán dương cái nào (0,0); rồi giảm dần qua từng
    # giá trị → bao gồm cả ngưỡng tại min (đoán dương tất → (1,1)).
    thresholds = [uniq[-1] + 1.0] + list(reversed(uniq))
    pts = [OperatingPoint(float(t), confusion_at(scores, labels, t))
           for t in thresholds]
    pts.sort(key=lambda p: (p.fpr, p.tpr))
    return pts


def auc(points) -> float:
    """Diện tích dưới ROC (hình thang). points: list OperatingPoint hoặc (fpr,tpr)."""
    xy = []
    for p in points:
        if isinstance(p, OperatingPoint):
            xy.append((p.fpr, p.tpr))
        else:
            xy.append((float(p[0]), float(p[1])))
    xy = sorted(set(xy))
    area = 0.0
    for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


# ---------- chọn điểm vận hành (= chọn ngưỡng tối ưu) ----------

def best_by_youden(points):
    """Youden's J = tpr - fpr lớn nhất (cân bằng bắt-được vs báo-nhầm)."""
    return max(points, key=lambda p: p.tpr - p.fpr) if points else None


def best_by_f1(points):
    return max(points, key=lambda p: p.confusion.f1) if points else None


def best_recall_at_fpr(points, max_fpr: float):
    """An toàn-trên-hết: recall (bắt được) cao nhất TRONG KHI báo nhầm ≤ max_fpr.
    Hợp với baby monitor: ghìm tỉ lệ báo nhầm dưới ngưỡng chịu được rồi tối đa độ nhạy."""
    ok = [p for p in points if p.fpr <= max_fpr]
    if not ok:
        return None
    # recall cao nhất; nếu hoà → fpr thấp hơn; nếu hoà nữa → ngưỡng cao hơn (chặt hơn)
    return max(ok, key=lambda p: (p.tpr, -p.fpr, p.threshold))
