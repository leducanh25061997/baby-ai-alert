"""HƯỚNG A — Bước 4: đo bộ phát hiện che trên dataset.csv đã trích.

Trả lời 2 câu hội đồng sẽ hỏi:
  1. "Nó tốt cỡ nào?"  → precision / recall / báo-nhầm + AUC.
  2. "Ngưỡng vote=2 có tối ưu không?" → quét ngưỡng + ROC + điểm vận hành tối ưu.

Score mặc định = max(nose_votes, mouth_votes) — đúng với rule hiện tại
(che nếu mũi≥2 HOẶC miệng≥2). Quét ngưỡng 0..4 cho biết nên đặt ngưỡng nào.

Chạy:
    python scripts/evaluate.py --csv dataset.csv
    python scripts/evaluate.py --csv dataset.csv --score sum --max-fpr 0.05 --plot

Chỉ phụ thuộc src/eval_metrics.py (thuần) + csv chuẩn. matplotlib là TUỲ CHỌN
(chỉ để xuất ROC.png; không có vẫn chạy + lưu roc.csv).
"""
import sys, csv, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_metrics import (
    confusion_at, sweep, roc_curve, auc,
    best_by_youden, best_by_f1, best_recall_at_fpr,
)


def load(csv_path, score_kind):
    scores, labels, current = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = row.get("label", "").strip()
            if lab not in ("0", "1"):
                continue
            nv = int(float(row["nose_votes"]))
            mv = int(float(row["mouth_votes"]))
            scores.append(nv + mv if score_kind == "sum" else max(nv, mv))
            labels.append(int(lab))
            oc = row.get("occluded_current", "")
            current.append(int(float(oc)) if oc not in ("", None) else None)
    return scores, labels, current


def fmt(c):
    return (f"P={c.precision:.3f} R={c.recall:.3f} "
            f"FPR={c.fpr:.3f} F1={c.f1:.3f} "
            f"(tp={c.tp} fp={c.fp} tn={c.tn} fn={c.fn})")


def main():
    ap = argparse.ArgumentParser(description="Đo detector + ROC từ dataset.csv")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--score", choices=["max", "sum"], default="max")
    ap.add_argument("--max-fpr", type=float, default=0.05,
                    help="Tran bao nham cho diem 'an toan nhat' (mac dinh 0.05)")
    ap.add_argument("--out-roc", default="roc.csv")
    ap.add_argument("--plot", action="store_true", help="Xuất roc.png (cần matplotlib)")
    args = ap.parse_args()

    if not Path(args.csv).is_file():
        sys.exit(f"❌ Không thấy {args.csv} — chạy extract_features.py trước.")
    scores, labels, current = load(args.csv, args.score)
    if not scores:
        sys.exit("❌ CSV không có dòng nhãn hợp lệ (label 0/1).")

    n_pos = sum(labels); n_neg = len(labels) - n_pos
    score_max = 8 if args.score == "sum" else 4
    print(f"\n=== DỮ LIỆU ===")
    print(f"  {len(labels)} frame | che(1)={n_pos}  không-che(0)={n_neg} "
          f"| score = {args.score}(votes) ∈ [0,{score_max}]")
    if n_pos == 0 or n_neg == 0:
        print("⚠ Cần CẢ 2 lớp (che + không che) để đo có ý nghĩa.")

    # 1) Rule hiện tại (vote >= 2)
    print(f"\n=== RULE HIỆN TẠI (score >= 2) ===")
    print("  " + fmt(confusion_at(scores, labels, 2)))
    if all(c is not None for c in current) and current:
        cc = confusion_at(current, labels, 1)  # occluded_current là 0/1 sẵn
        print("  (theo cờ occluded_current đã ghi: " + fmt(cc) + ")")

    # 2) Quét ngưỡng vote
    print(f"\n=== QUÉT NGƯỠNG (chọn điểm vận hành) ===")
    pts = sweep(scores, labels, list(range(0, score_max + 1)))
    print("  thr | precision recall   FPR    F1")
    for p in pts:
        c = p.confusion
        print(f"  {int(p.threshold):>3} |   {c.precision:.3f}   {c.recall:.3f}  "
              f"{c.fpr:.3f}  {c.f1:.3f}")

    # 3) ROC + AUC
    roc = roc_curve(scores, labels)
    area = auc(roc)
    print(f"\n=== ROC / AUC ===")
    print(f"  AUC = {area:.4f}   "
          f"({'xuất sắc' if area>=0.9 else 'tốt' if area>=0.8 else 'khá' if area>=0.7 else 'yếu'})")

    # 4) Điểm vận hành tối ưu theo 3 tiêu chí
    print(f"\n=== NGƯỠNG TỐI ƯU GỢI Ý ===")
    y = best_by_youden(pts); f = best_by_f1(pts)
    s = best_recall_at_fpr(pts, args.max_fpr)
    print(f"  Youden (cân bằng)      : thr={int(y.threshold)}  {fmt(y.confusion)}")
    print(f"  F1 (cân bằng P/R)      : thr={int(f.threshold)}  {fmt(f.confusion)}")
    if s is not None:
        print(f"  An toàn (FPR≤{args.max_fpr:.0%}, max recall): thr={int(s.threshold)}  {fmt(s.confusion)}")
    else:
        print(f"  An toàn (FPR≤{args.max_fpr:.0%}): KHÔNG có ngưỡng nào đạt trần báo nhầm này.")

    # 5) Lưu roc.csv
    with open(args.out_roc, "w", newline="", encoding="utf-8") as f2:
        wr = csv.writer(f2)
        wr.writerow(["threshold", "fpr", "tpr", "precision", "f1"])
        for p in roc:
            c = p.confusion
            wr.writerow([p.threshold, f"{c.fpr:.4f}", f"{c.recall:.4f}",
                         f"{c.precision:.4f}", f"{c.f1:.4f}"])
    print(f"\n💾 Lưu đường ROC → {args.out_roc}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [p.fpr for p in roc]; ys = [p.tpr for p in roc]
            plt.figure(figsize=(5, 5))
            plt.plot(xs, ys, "-o", label=f"AUC={area:.3f}")
            plt.plot([0, 1], [0, 1], "--", color="gray", label="ngẫu nhiên")
            plt.xlabel("False-alarm rate (FPR)"); plt.ylabel("Recall (TPR)")
            plt.title("ROC — bộ phát hiện che"); plt.legend(); plt.grid(True, alpha=.3)
            plt.savefig("roc.png", dpi=120, bbox_inches="tight")
            print("🖼  Lưu roc.png")
        except ImportError:
            print("⚠ Không có matplotlib → bỏ qua roc.png (roc.csv vẫn đã lưu).")


if __name__ == "__main__":
    main()
