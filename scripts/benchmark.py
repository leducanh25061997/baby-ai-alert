"""Benchmark pipeline — đo FPS thực tế của từng stage để biết bottleneck.

CPU-only: project không hỗ trợ NVIDIA GPU; Raspberry Pi 4 không có NPU/CUDA.

Đo riêng:
  1. Camera read (raw)
  2. MediaPipe FaceMesh
  3. Histogram extraction + correlation
  4. YOLO inference (CPU)
  5. Tổng end-to-end

Sử dụng:
  python scripts/benchmark.py                       # webcam mặc định
  python scripts/benchmark.py --source 0 --frames 200
  python scripts/benchmark.py --no-yolo             # chỉ đo face_mesh + histogram
  python scripts/benchmark.py --synthetic           # dùng frame giả nếu không có camera
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def make_synthetic_frame(w=1280, h=720):
    """Tạo frame giả có 'mặt' (vòng tròn da-tone) để mediapipe có cái để chạy."""
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), 180, (180, 160, 140), -1)  # khuôn mặt
    cv2.circle(img, (w // 2 - 60, h // 2 - 30), 12, (40, 40, 40), -1)  # mắt T
    cv2.circle(img, (w // 2 + 60, h // 2 - 30), 12, (40, 40, 40), -1)  # mắt P
    cv2.ellipse(img, (w // 2, h // 2 + 50), (40, 15), 0, 0, 360, (100, 50, 50), -1)
    return img


def open_camera(source, w, h, fps):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def fmt_stats(name, samples_ms):
    if not samples_ms:
        return f"  {name:<22}: (no data)"
    avg  = statistics.mean(samples_ms)
    med  = statistics.median(samples_ms)
    p95  = statistics.quantiles(samples_ms, n=20)[18] if len(samples_ms) >= 20 else max(samples_ms)
    fps  = 1000.0 / avg if avg > 0 else 0.0
    return f"  {name:<22}: avg={avg:6.2f}ms  median={med:6.2f}ms  p95={p95:6.2f}ms  → {fps:5.1f} FPS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",      default="0")
    ap.add_argument("--width",       type=int, default=640)
    ap.add_argument("--height",      type=int, default=480)
    ap.add_argument("--fps",         type=int, default=15)
    ap.add_argument("--frames",      type=int, default=150, help="Số frame benchmark")
    ap.add_argument("--warmup",      type=int, default=10)
    ap.add_argument("--no-yolo",     action="store_true")
    ap.add_argument("--synthetic",   action="store_true",
                    help="dùng frame giả thay vì camera (debug pipeline)")
    args = ap.parse_args()

    print("Device: cpu (project CPU-only, không hỗ trợ NVIDIA GPU)")
    print(f"OpenCV: {cv2.__version__}")
    print()

    # ----- Open camera or synthetic -----
    cap = None
    if not args.synthetic:
        cap = open_camera(args.source, args.width, args.height, args.fps)
        if cap is None:
            print(f"❌ Không mở được camera {args.source}, fallback sang synthetic")
            args.synthetic = True

    if args.synthetic:
        print(f"Dùng synthetic frame {args.width}x{args.height}")
        synthetic = make_synthetic_frame(args.width, args.height)

    # ----- Load mediapipe -----
    try:
        import mediapipe as mp
    except ImportError:
        print("❌ mediapipe chưa cài. Chạy: pip install mediapipe")
        return 1
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    # ----- Load YOLO (CPU-only) -----
    yolo = None
    yolo_device = "cpu"
    if not args.no_yolo:
        model_path = ROOT / "yolov8n.pt"
        if not model_path.exists():
            print(f"⚠️  Không thấy {model_path}, bỏ YOLO")
        else:
            try:
                from ultralytics import YOLO
                yolo = YOLO(str(model_path))
                yolo.to(yolo_device)
                print(f"YOLO: {model_path.name} on {yolo_device}")
            except ImportError:
                print("⚠️  ultralytics chưa cài, bỏ YOLO")
            except Exception as e:
                print(f"⚠️  Lỗi load YOLO: {e}")

    print(f"Warmup {args.warmup} frame, đo {args.frames} frame...\n")

    # ----- Histogram helpers (tái tạo cho gọn, không import từ main) -----
    def calc_hist(patch):
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [36, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    baseline_hist = None
    times_read    = []
    times_face    = []
    times_hist    = []
    times_yolo    = []
    times_total   = []

    for i in range(args.warmup + args.frames):
        is_warmup = i < args.warmup

        # 1) Camera read
        t0 = time.perf_counter()
        if args.synthetic:
            frame = synthetic.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                continue
        t_read = (time.perf_counter() - t0) * 1000

        h, w = frame.shape[:2]

        # 2) MediaPipe
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        t_face = (time.perf_counter() - t0) * 1000

        # 3) Histogram (chỉ chạy khi có face)
        t_hist = 0.0
        if res.multi_face_landmarks:
            t0 = time.perf_counter()
            lms = res.multi_face_landmarks[0].landmark
            for idx in (4, 14):
                lm = lms[idx]
                x = int(np.clip(lm.x * w, 0, w - 1))
                y = int(np.clip(lm.y * h, 0, h - 1))
                patch = frame[max(0, y-25):min(h, y+25), max(0, x-25):min(w, x+25)]
                if patch.size:
                    patch = cv2.resize(patch, (50, 50))
                    hist = calc_hist(patch)
                    if baseline_hist is None:
                        baseline_hist = hist
                    cv2.compareHist(hist.astype(np.float32),
                                    baseline_hist.astype(np.float32),
                                    cv2.HISTCMP_CORREL)
            t_hist = (time.perf_counter() - t0) * 1000

        # 4) YOLO
        t_yolo = 0.0
        if yolo is not None:
            t0 = time.perf_counter()
            try:
                yolo(frame, classes=[0], conf=0.35,
                     device=yolo_device, verbose=False)
            except Exception:
                pass
            t_yolo = (time.perf_counter() - t0) * 1000

        t_total = t_read + t_face + t_hist + t_yolo

        if not is_warmup:
            times_read.append(t_read)
            times_face.append(t_face)
            times_hist.append(t_hist)
            if yolo is not None:
                times_yolo.append(t_yolo)
            times_total.append(t_total)

        if i % 20 == 0:
            print(f"  frame {i:>3}/{args.warmup + args.frames}  total={t_total:.1f}ms")

    if cap:
        cap.release()
    face_mesh.close()

    print("\n===== KẾT QUẢ =====")
    print(fmt_stats("Camera read",    times_read))
    print(fmt_stats("MediaPipe",      times_face))
    print(fmt_stats("Histogram",      [t for t in times_hist if t > 0]))
    if yolo is not None:
        print(fmt_stats(f"YOLO ({yolo_device})", times_yolo))
    print(fmt_stats("END-TO-END",     times_total))

    if times_total:
        avg_total = statistics.mean(times_total)
        target_ms = 1000.0 / args.fps
        print()
        if avg_total <= target_ms:
            print(f"✅ Pipeline {avg_total:.1f}ms ≤ ngân sách {target_ms:.1f}ms ({args.fps} FPS) → real-time OK")
        else:
            achievable = 1000.0 / avg_total
            print(f"⚠️  Pipeline {avg_total:.1f}ms > ngân sách {target_ms:.1f}ms → "
                  f"thực tế chỉ đạt ~{achievable:.0f} FPS")
            print("   Khuyến nghị: giảm camera FPS xuống bằng giá trị này, hoặc:")
            if yolo is not None and times_yolo and statistics.mean(times_yolo) > 30:
                print("   - YOLO chiếm > 30ms → tăng YOLO_EVERY (vd 15), hoặc gắn Coral USB TPU")
            elif times_face and statistics.mean(times_face) > 25:
                print("   - MediaPipe chậm → giảm resolution xuống 480x360")
    return 0


if __name__ == "__main__":
    sys.exit(main())
