"""HƯỚNG A — Bước 3: trích đặc trưng từ video ĐÃ GÁN NHÃN → dataset.csv.

Chạy detector thật (MediaPipe + OcclusionDetector) qua từng clip, mỗi frame
THẤY MẶT xuất 1 dòng: tín hiệu thô + phiếu + quyết định hiện tại + đặc trưng
TƯƠNG ĐỐI (cho hướng B sau này) + nhãn.

Cấu trúc thư mục (1 session = 1 bối cảnh: 1 kiểu sáng / vị trí / đối tượng):

    dataset/
      session01_sang/
        calib.mp4         # mặt/búp bê RÕ, không che → học baseline cho session
        sach_01.mp4       # KHÔNG che  → nhãn 0
        che_tay_01.mp4    # tay che    → nhãn 1
        che_chan_01.mp4   # chăn che   → nhãn 1
      session02_toi/
        calib.mp4
        ...

Quy ước nhãn theo TÊN FILE: bắt đầu 'calib' = clip hiệu chỉnh (không xuất);
'che*' = 1 ; 'sach*'/'clean*' = 0 ; khác → bỏ qua kèm cảnh báo.

Chạy:
    python scripts/extract_features.py --dataset dataset --out dataset.csv

Lưu ý: script này là CÔNG CỤ DEV (cần cv2 + mediapipe), chạy trên máy bạn / Pi —
KHÔNG phải code chạy production. Logic ĐO nằm ở src/eval_metrics.py (test riêng).
"""
import sys, csv, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import mediapipe as mp
from occlusion_detector import OcclusionDetector
from scene_monitor import SceneMonitor

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

CSV_FIELDS = [
    "session", "clip", "label", "frame_idx",
    "nose_hist_corr", "mouth_hist_corr",
    "nose_skin_ratio", "mouth_skin_ratio",
    "nose_votes", "mouth_votes", "occluded_current",
    # Đặc trưng TƯƠNG ĐỐI so với baseline (generalize tốt — dùng cho hướng B)
    "nose_skin_rel", "mouth_skin_rel",
]


def label_of(name: str):
    """None = calib (bỏ qua xuất); 1 = che; 0 = không che; -1 = không nhận ra."""
    n = name.lower()
    if n.startswith("calib"):
        return None
    if n.startswith("che"):
        return 1
    if n.startswith("sach") or n.startswith("clean"):
        return 0
    return -1


def _rel(cur, base):
    return float(cur) / float(base) if base else 0.0


def calibrate(detector, scene, facemesh, calib_path, calib_sec):
    """Học baseline từ clip hiệu chỉnh. Trả (ok, message)."""
    cap = cv2.VideoCapture(str(calib_path))
    if not cap.isOpened():
        return False, f"không mở được {calib_path.name}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(calib_sec * fps) if calib_sec > 0 else 10 ** 9
    n = 0
    while n < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        h, w = frame.shape[:2]
        scene.analyze(frame)
        res = facemesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            detector.add_calibration_sample(frame, lms, w, h)
            scene.update_sharp_baseline()
    cap.release()
    return detector.finalize_calibration()


def extract_clip(detector, scene, facemesh, clip_path, label, session, writer):
    """Chạy detector qua 1 clip, ghi 1 dòng/frame-thấy-mặt. Trả (n_frames, n_written)."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"   ⚠ không mở được {clip_path.name}, bỏ qua")
        return 0, 0
    idx = written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        h, w = frame.shape[:2]
        scene.analyze(frame)
        res = facemesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue  # mất mặt → đường face_lost xử lý, không thuộc bộ phát hiện che
        lms = res.multi_face_landmarks[0].landmark
        cr = detector.check(frame, lms, w, h, prev_in_alert=False)
        if cr is None:
            continue
        nb, mb = detector.nose, detector.mouth
        ns_rel = _rel(cr.nose_skin_ratio,  nb.skin_ratio)
        ms_rel = _rel(cr.mouth_skin_ratio, mb.skin_ratio)
        writer.writerow({
            "session": session, "clip": clip_path.name, "label": label,
            "frame_idx": idx,
            "nose_hist_corr": f"{cr.nose_hist_corr:.4f}",
            "mouth_hist_corr": f"{cr.mouth_hist_corr:.4f}",
            "nose_skin_ratio": f"{cr.nose_skin_ratio:.4f}",
            "mouth_skin_ratio": f"{cr.mouth_skin_ratio:.4f}",
            "nose_votes": cr.nose_votes_for_occluded,
            "mouth_votes": cr.mouth_votes_for_occluded,
            "occluded_current": int(cr.occluded),
            "nose_skin_rel": f"{ns_rel:.4f}", "mouth_skin_rel": f"{ms_rel:.4f}",
        })
        written += 1
    cap.release()
    return idx, written


def main():
    ap = argparse.ArgumentParser(description="Trích đặc trưng từ video gán nhãn → CSV")
    ap.add_argument("--dataset", required=True, help="Thư mục dataset/ chứa các session")
    ap.add_argument("--out", default="dataset.csv", help="File CSV xuất ra")
    ap.add_argument("--calib-sec", type=float, default=5.0,
                    help="Số giây đầu của calib.mp4 dùng học baseline (0 = cả clip)")
    ap.add_argument("--mp-conf", type=float, default=0.6, help="MediaPipe min confidence")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        sys.exit(f"❌ Không thấy thư mục dataset: {dataset}")
    sessions = sorted(p for p in dataset.iterdir() if p.is_dir())
    if not sessions:
        sys.exit(f"❌ {dataset} không có session con nào (mỗi session là 1 thư mục).")

    mp_face = mp.solutions.face_mesh
    tot_written = 0
    skipped_clips = []
    with open(args.out, "w", newline="", encoding="utf-8") as f, \
         mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                          min_detection_confidence=args.mp_conf,
                          min_tracking_confidence=args.mp_conf) as facemesh:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for sess in sessions:
            clips = [p for p in sorted(sess.iterdir())
                     if p.suffix.lower() in VIDEO_EXTS]
            calib = next((p for p in clips if label_of(p.stem) is None), None)
            if calib is None:
                print(f"⚠ Session '{sess.name}' THIẾU clip calib*.mp4 → bỏ qua.")
                continue
            detector = OcclusionDetector()
            scene = SceneMonitor()
            ok, msg = calibrate(detector, scene, facemesh, calib, args.calib_sec)
            if not ok:
                print(f"⚠ Session '{sess.name}' calibrate FAIL: {msg} → bỏ qua.")
                continue
            print(f"📂 {sess.name}: calib OK (q={detector.calibration_quality:.2f})")
            for clip in clips:
                lab = label_of(clip.stem)
                if lab is None:
                    continue          # calib
                if lab == -1:
                    skipped_clips.append(f"{sess.name}/{clip.name}")
                    continue
                n, w = extract_clip(detector, scene, facemesh, clip, lab, sess.name, writer)
                tot_written += w
                tag = "CHE" if lab == 1 else "sach"
                print(f"   - {clip.name:30s} [{tag}] {w}/{n} frame có mặt")

    print(f"\n✅ Xong: {tot_written} dòng → {args.out}")
    if skipped_clips:
        print(f"⚠ Bỏ qua {len(skipped_clips)} clip không nhận ra nhãn "
              f"(tên không bắt đầu bằng che/sach/clean/calib):")
        for c in skipped_clips:
            print(f"     {c}")
    if tot_written == 0:
        print("⚠ KHÔNG xuất được dòng nào — kiểm tra clip có thấy mặt không, "
              "calib có đạt không.")


if __name__ == "__main__":
    main()
