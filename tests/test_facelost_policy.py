"""Test logic THUẦN cho nhánh FACE_LOST (facelost_policy). Thuần stdlib.

Chạy: python tests/test_facelost_policy.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from facelost_policy import (
    estimate_yaw_proxy, classify_face_loss, face_lost_severity,
    MANNER_TURNED, MANNER_COVERED, SEVERITY_SOFT, SEVERITY_LOUD,
    DEFAULT_PROFILE_YAW,
)


def test_yaw_frontal_is_zero():
    """Mũi nằm chính giữa 2 mép mặt → yaw ~0 (chính diện)."""
    assert abs(estimate_yaw_proxy(0.5, 0.4, 0.6)) < 1e-9
    print("✅ test_yaw_frontal_is_zero")


def test_yaw_turned_positive():
    """Mũi lệch về mép phải → yaw dương lớn."""
    y = estimate_yaw_proxy(0.58, 0.40, 0.60)   # center .5 half .1 → .8
    assert abs(y - 0.8) < 1e-9, y
    print("✅ test_yaw_turned_positive")


def test_yaw_clamped():
    """Mũi lệch quá biên → yaw bị kẹp trong [-1, 1]."""
    assert estimate_yaw_proxy(0.95, 0.40, 0.60) == 1.0
    assert estimate_yaw_proxy(0.05, 0.40, 0.60) == -1.0
    print("✅ test_yaw_clamped")


def test_yaw_degenerate_edges():
    """2 mép trùng nhau (suy biến) → 0.0, không chia 0."""
    assert estimate_yaw_proxy(0.5, 0.5, 0.5) == 0.0
    print("✅ test_yaw_degenerate_edges")


def test_classify_none_is_covered():
    """Không có dữ liệu yaw gần → mặc định COVERED (an-toàn-trước)."""
    assert classify_face_loss(None) == MANNER_COVERED
    print("✅ test_classify_none_is_covered")


def test_classify_profile_is_turned():
    """|yaw| lớn (đang xoay nghiêng) → TURNED."""
    assert classify_face_loss(0.8) == MANNER_TURNED
    assert classify_face_loss(DEFAULT_PROFILE_YAW) == MANNER_TURNED  # biên: >=
    print("✅ test_classify_profile_is_turned")


def test_classify_frontal_is_covered():
    """|yaw| nhỏ (còn chính diện) mà mất mặt → COVERED (nghi che)."""
    assert classify_face_loss(0.2) == MANNER_COVERED
    print("✅ test_classify_frontal_is_covered")


def test_severity_covered_always_loud():
    """COVERED → báo to ngay cả khi mới mất mặt."""
    assert face_lost_severity(MANNER_COVERED, elapsed=0.0, escalate_sec=30) == SEVERITY_LOUD
    assert face_lost_severity(MANNER_COVERED, elapsed=99.0, escalate_sec=30) == SEVERITY_LOUD
    print("✅ test_severity_covered_always_loud")


def test_severity_turned_soft_then_loud():
    """TURNED → nhẹ trước, leo thang lên to khi kéo dài ≥ escalate_sec."""
    assert face_lost_severity(MANNER_TURNED, elapsed=5.0,  escalate_sec=30) == SEVERITY_SOFT
    assert face_lost_severity(MANNER_TURNED, elapsed=29.9, escalate_sec=30) == SEVERITY_SOFT
    assert face_lost_severity(MANNER_TURNED, elapsed=30.0, escalate_sec=30) == SEVERITY_LOUD
    assert face_lost_severity(MANNER_TURNED, elapsed=45.0, escalate_sec=30) == SEVERITY_LOUD
    print("✅ test_severity_turned_soft_then_loud")


if __name__ == "__main__":
    tests = [
        test_yaw_frontal_is_zero,
        test_yaw_turned_positive,
        test_yaw_clamped,
        test_yaw_degenerate_edges,
        test_classify_none_is_covered,
        test_classify_profile_is_turned,
        test_classify_frontal_is_covered,
        test_severity_covered_always_loud,
        test_severity_turned_soft_then_loud,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"❌ {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
    total = len(tests)
    if failed == 0:
        print(f"\n🎉 {total}/{total} test PASS")
    else:
        print(f"\n❌ {failed}/{total} FAIL")
        sys.exit(1)
