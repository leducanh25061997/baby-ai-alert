"""Test eval_metrics — logic đo thuần (không cần cv2/mediapipe/video).

Chạy: python tests/test_eval_metrics.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_metrics import (
    Confusion, confusion_at, sweep, roc_curve, auc,
    best_by_youden, best_by_f1, best_recall_at_fpr,
)


def approx(a, b, eps=1e-9):
    return abs(a - b) <= eps


# ============== Confusion ==============

def test_confusion_basic_metrics():
    c = Confusion(tp=8, fp=2, tn=18, fn=2)
    assert approx(c.precision, 8/10)
    assert approx(c.recall,    8/10)
    assert approx(c.fpr,       2/20)
    assert approx(c.specificity, 18/20)
    assert approx(c.accuracy, 26/30)
    assert approx(c.f1, 2*0.8*0.8/(0.8+0.8))
    print("✅ test_confusion_basic_metrics")


def test_confusion_zero_division_safe():
    c = Confusion(tp=0, fp=0, tn=0, fn=0)
    assert c.precision == 0.0 and c.recall == 0.0 and c.fpr == 0.0
    assert c.f1 == 0.0 and c.accuracy == 0.0
    print("✅ test_confusion_zero_division_safe")


# ============== confusion_at ==============

def test_confusion_at_threshold():
    # score = max votes (0..4), label 1=che
    scores = [0, 1, 2, 3, 4]
    labels = [0, 0, 1, 1, 1]
    # threshold=2 (đúng rule hiện tại): đoán dương khi score>=2
    c = confusion_at(scores, labels, 2)
    assert (c.tp, c.fp, c.tn, c.fn) == (3, 0, 2, 0)
    assert approx(c.recall, 1.0) and approx(c.fpr, 0.0)
    # threshold=1: score 1 (label 0) thành FP
    c1 = confusion_at(scores, labels, 1)
    assert (c1.tp, c1.fp, c1.tn, c1.fn) == (3, 1, 1, 0)
    print("✅ test_confusion_at_threshold")


# ============== ROC / AUC ==============

def test_roc_perfect_separation_auc_1():
    scores = [0.0, 0.1, 0.9, 1.0]
    labels = [0,   0,   1,   1]
    pts = roc_curve(scores, labels)
    # phải chứa điểm (fpr=0, tpr=1) → phân tách hoàn hảo
    assert any(approx(p.fpr, 0.0) and approx(p.tpr, 1.0) for p in pts)
    assert approx(auc(pts), 1.0)
    print("✅ test_roc_perfect_separation_auc_1")


def test_roc_no_discrimination_auc_half():
    # mọi score bằng nhau → không phân biệt được che/không → AUC = 0.5
    scores = [1, 1, 1, 1]
    labels = [0, 1, 0, 1]
    pts = roc_curve(scores, labels)
    assert approx(auc(pts), 0.5)
    print(f"✅ test_roc_no_discrimination_auc_half  (auc={auc(pts):.3f})")


def test_roc_partial_separation_auc():
    # [0,1,2,3]/[0,1,0,1]: positives={1,3} > negatives={0,2} ở 3/4 cặp → AUC=0.75
    scores = [0, 1, 2, 3]
    labels = [0, 1, 0, 1]
    pts = roc_curve(scores, labels)
    assert approx(auc(pts), 0.75), f"auc={auc(pts)}"
    print(f"✅ test_roc_partial_separation_auc  (auc={auc(pts):.3f})")


def test_roc_endpoints_present():
    scores = [0, 1, 2, 3, 4]
    labels = [0, 0, 1, 1, 1]
    pts = roc_curve(scores, labels)
    assert any(approx(p.fpr, 0.0) and approx(p.tpr, 0.0) for p in pts), "thiếu (0,0)"
    assert any(approx(p.fpr, 1.0) and approx(p.tpr, 1.0) for p in pts), "thiếu (1,1)"
    print("✅ test_roc_endpoints_present")


# ============== sweep + chọn ngưỡng ==============

def test_sweep_vote_thresholds():
    scores = [0, 1, 2, 3, 4, 2, 1, 0]
    labels = [0, 0, 1, 1, 1, 1, 0, 0]
    pts = sweep(scores, labels, [0, 1, 2, 3, 4])
    assert len(pts) == 5
    by_t = {p.threshold: p for p in pts}
    # threshold=2: che thật = {2,3,4,2} đều >=2 → tp=4; không che {0,1,1,0} <2 → tn=4
    c2 = by_t[2.0].confusion
    assert (c2.tp, c2.fp, c2.tn, c2.fn) == (4, 0, 4, 0)
    print("✅ test_sweep_vote_thresholds")


def test_best_recall_at_fpr_safety_first():
    # điểm A: tpr=0.9 fpr=0.02 ; điểm B: tpr=0.98 fpr=0.20
    # với trần báo nhầm 5% → phải chọn A (B vượt trần dù recall cao hơn)
    scores = [0, 1, 2, 3]
    labels = [0, 0, 1, 1]
    pts = roc_curve(scores, labels)
    chosen = best_recall_at_fpr(pts, max_fpr=0.05)
    assert chosen is not None and chosen.fpr <= 0.05
    print("✅ test_best_recall_at_fpr_safety_first")


def test_best_recall_at_fpr_none_when_all_exceed():
    # mọi điểm có fpr>0 trừ (0,0); trần fpr=-1 → không điểm nào đạt
    scores = [0, 1]
    labels = [0, 1]
    pts = roc_curve(scores, labels)
    assert best_recall_at_fpr(pts, max_fpr=-1.0) is None
    print("✅ test_best_recall_at_fpr_none_when_all_exceed")


def test_best_by_youden_and_f1_run():
    scores = [0, 1, 2, 3, 4]
    labels = [0, 0, 1, 1, 1]
    pts = roc_curve(scores, labels)
    y = best_by_youden(pts)
    assert approx(y.tpr - y.fpr, 1.0)   # phân tách hoàn hảo → J=1
    f = best_by_f1(sweep(scores, labels, [0, 1, 2, 3, 4]))
    assert approx(f.confusion.f1, 1.0)
    print("✅ test_best_by_youden_and_f1_run")


# ============== runner ==============

if __name__ == "__main__":
    tests = [
        test_confusion_basic_metrics,
        test_confusion_zero_division_safe,
        test_confusion_at_threshold,
        test_roc_perfect_separation_auc_1,
        test_roc_no_discrimination_auc_half,
        test_roc_partial_separation_auc,
        test_roc_endpoints_present,
        test_sweep_vote_thresholds,
        test_best_recall_at_fpr_safety_first,
        test_best_recall_at_fpr_none_when_all_exceed,
        test_best_by_youden_and_f1_run,
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
