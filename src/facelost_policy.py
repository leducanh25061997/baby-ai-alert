"""Logic THUẦN cho nhánh phát hiện 'mất mặt' (FACE_LOST) — phân biệt
'bé quay đầu/nằm nghiêng (an toàn)' với 'mặt bị vùi/che kín (nguy hiểm)'.

Tách khỏi main.py để unit-test độc lập (giống state_machine.py / alert_policy.py):
KHÔNG phụ thuộc cv2/mediapipe/telegram — chỉ nhận số (toạ độ landmark đã chuẩn
hóa, |yaw|, elapsed) và trả về phân loại/mức độ.

Vai trò trong hệ thống nhiều lớp giảm báo nhầm khi mất mặt:
  - Lớp 1 (BlazeFace, ở main.py): mặt nghiêng vẫn detect được → không tính mất mặt.
  - Lớp 3 (module này): khi CẢ FaceMesh lẫn BlazeFace đều mất mặt, dùng 'cách biến
    mất' (head-pose ngay trước khi mất) để đoán nguyên nhân:
      * mất mặt trong lúc đang xoay sang nghiêng → MANNER_TURNED (thiên về an toàn)
      * mất mặt khi còn gần chính diện     → MANNER_COVERED (nghi bị che đột ngột)
  - Lớp 2 (module này): chọn mức độ cảnh báo (nhẹ → to) theo manner + thời gian.
"""
from __future__ import annotations

# Ngưỡng |yaw proxy| coi là 'đang xoay sang nghiêng'. yaw proxy ~0 = chính diện,
# tiến tới ±1 khi mặt xoay hẳn sang một bên.
DEFAULT_PROFILE_YAW = 0.55

MANNER_TURNED  = "turned"   # mất mặt trong lúc đang xoay nghiêng → thiên về an toàn
MANNER_COVERED = "covered"  # mất mặt khi còn gần chính diện → nghi bị che đột ngột

SEVERITY_SOFT = "soft"      # nhắc nhẹ "kiểm tra giúp"
SEVERITY_LOUD = "loud"      # cảnh báo nguy hiểm "kiểm tra ngay"


def estimate_yaw_proxy(nose_x: float, left_x: float, right_x: float) -> float:
    """Proxy yaw chuẩn hóa từ toạ độ x (đã chuẩn hóa 0..1) của mũi và 2 mép mặt.

    Chính diện: mũi nằm giữa 2 mép → ~0. Mặt xoay sang phải/trái → mũi lệch về
    một mép → tiến tới ±1. Bền với vị trí/kích thước mặt trong khung (vì chuẩn
    hóa theo bề ngang mặt). Trả 0.0 nếu 2 mép trùng nhau (suy biến)."""
    center = (left_x + right_x) / 2.0
    half = abs(right_x - left_x) / 2.0
    if half <= 1e-6:
        return 0.0
    val = (nose_x - center) / half
    return max(-1.0, min(1.0, val))   # kẹp [-1, 1]


def classify_face_loss(last_abs_yaw, profile_yaw: float = DEFAULT_PROFILE_YAW) -> str:
    """Phân loại 'cách biến mất' dựa vào |yaw| ngay TRƯỚC khi mất mặt.

    last_abs_yaw=None (không có dữ liệu yaw gần đây — vd mặt mất đã lâu) → mặc
    định MANNER_COVERED: an-toàn-trước, coi như nghi bị che để báo to."""
    if last_abs_yaw is None:
        return MANNER_COVERED
    return MANNER_TURNED if last_abs_yaw >= profile_yaw else MANNER_COVERED


def face_lost_severity(manner: str, elapsed: float, escalate_sec: float) -> str:
    """Mức độ cảnh báo cho FACE_LOST (Lớp 2 — leo thang):

      - manner=COVERED (nghi che đột ngột) → LUÔN báo to ngay từ đầu.
      - manner=TURNED  (đang nằm nghiêng)  → nhắc NHẸ trước; chỉ leo thang lên
        báo to khi tình trạng kéo dài ≥ escalate_sec (lúc đó 'nằm nghiêng' quá
        lâu cũng đáng ngại → an-toàn-trước)."""
    if manner == MANNER_COVERED:
        return SEVERITY_LOUD
    return SEVERITY_LOUD if elapsed >= escalate_sec else SEVERITY_SOFT
