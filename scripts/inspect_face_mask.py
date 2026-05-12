"""Inspect face_mask.pt — phát hiện model trained cho task gì + class nào.

Chạy:
    python scripts/inspect_face_mask.py

Output sẽ giúp quyết định có tích hợp model này làm primary detector hay không.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "face_mask.pt"


def main():
    if not MODEL_PATH.exists():
        print(f"❌ Không thấy {MODEL_PATH}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"❌ Chưa cài ultralytics: {e}")
        print("   Chạy: pip install ultralytics")
        return 1

    print(f"📦 Loading model: {MODEL_PATH}")
    print(f"   File size: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    try:
        m = YOLO(str(MODEL_PATH))
    except Exception as e:
        print(f"❌ Không load được: {e}")
        print(f"   Có thể không phải YOLO format. Thử torch.load trực tiếp:")
        try:
            import torch
            ckpt = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
            print(f"   Type: {type(ckpt).__name__}")
            if isinstance(ckpt, dict):
                print(f"   Keys: {list(ckpt.keys())[:20]}")
                for k in ('model_type', 'architecture', 'classes', 'names'):
                    if k in ckpt:
                        print(f"   {k}: {ckpt[k]}")
        except Exception as e2:
            print(f"   torch.load cũng fail: {e2}")
        return 1

    print("=" * 60)
    print("✅ Model loaded — quan trọng nhất CHÍNH LÀ phần dưới đây:")
    print("=" * 60)

    print(f"\n🎯 Task              : {m.task}")
    print(f"📋 Classes           : {m.names}")
    print(f"📊 Số class          : {len(m.names) if m.names else '?'}")

    try:
        n_params = sum(p.numel() for p in m.model.parameters())
        print(f"🔢 Số parameters     : {n_params:,}  (~{n_params*4/1024/1024:.1f}MB FP32)")
    except Exception:
        pass

    try:
        print(f"📐 Default imgsz     : {m.model.args.get('imgsz', 'unknown')}")
    except Exception:
        pass

    # Test inference speed
    print("\n=== Test inference speed ===")
    try:
        import numpy as np
        for size in [320, 640]:
            test_img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            # Warmup
            m(test_img, verbose=False)
            # Measure
            t0 = time.perf_counter()
            n = 10
            for _ in range(n):
                m(test_img, verbose=False)
            avg_ms = (time.perf_counter() - t0) / n * 1000
            print(f"   {size}x{size}: {avg_ms:.1f} ms/frame  (~{1000/avg_ms:.0f} FPS)")
    except Exception as e:
        print(f"   Skipped: {e}")

    # Try real frame from webcam if available
    print("\n=== Test trên webcam (nếu có) ===")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                results = m(frame, verbose=False)
                for r in results:
                    n_boxes = len(r.boxes) if r.boxes is not None else 0
                    print(f"   Phát hiện {n_boxes} object")
                    if n_boxes > 0:
                        for i in range(n_boxes):
                            cls = int(r.boxes.cls[i].item())
                            conf = float(r.boxes.conf[i].item())
                            print(f"     - {m.names.get(cls, cls)}: {conf:.2f}")
            cap.release()
        else:
            print("   Webcam không mở được, skip.")
    except Exception as e:
        print(f"   Skipped: {e}")

    print("\n=" * 30)
    print("📤 Gửi toàn bộ output trên (đặc biệt phần Classes) để mình đánh giá:")
    print("   - Nếu Classes có {'face_visible', 'face_covered'} hoặc tương tự → ✅ tích hợp được")
    print("   - Nếu Classes là {'mask', 'no_mask'} (COVID-style) → ⚠️ tích hợp được nhưng cần test")
    print("   - Nếu Classes không liên quan (vd 'person', 'dog'...) → ❌ không dùng được, train mới")
    return 0


if __name__ == "__main__":
    sys.exit(main())
