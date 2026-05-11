"""Standalone webcam verifier — chạy ĐỘC LẬP, không phụ thuộc mediapipe/torch.

Mục đích: trước khi cài cả pipeline, verify Logitech webcam (hoặc camera khác)
hoạt động bình thường: mở được, đúng resolution, đủ FPS, không drop frame.

Sử dụng:
  python scripts/test_webcam.py                      # camera mặc định 0
  python scripts/test_webcam.py --source 1           # camera USB index 1
  python scripts/test_webcam.py --source rtsp://...  # IP camera
  python scripts/test_webcam.py --headless           # không hiện cửa sổ, chỉ đo FPS
  python scripts/test_webcam.py --duration 10        # đo trong 10 giây
"""
import argparse
import sys
import time
from pathlib import Path

import cv2


def open_camera(source, width, height, fps):
    """Mở camera với MJPG + buffer=1. Trả về (cap, actual_w, actual_h, actual_fps)."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None, 0, 0, 0.0
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return (cap,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS))


def scan_cameras(max_index=4):
    """Liệt kê các camera index khả dụng."""
    print("Đang scan camera index 0..{}...".format(max_index))
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  [OK]   index {i} → {w}x{h}")
                found.append(i)
            else:
                print(f"  [open nhung khong read duoc] index {i}")
            cap.release()
    if not found:
        print("  Khong tim thay camera nao.")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",   default="0", help="USB index hoặc RTSP URL")
    ap.add_argument("--width",    type=int, default=1280)
    ap.add_argument("--height",   type=int, default=720)
    ap.add_argument("--fps",      type=int, default=30)
    ap.add_argument("--duration", type=int, default=8, help="thời gian đo (giây)")
    ap.add_argument("--headless", action="store_true", help="không mở cửa sổ")
    ap.add_argument("--scan",     action="store_true", help="liệt kê các camera khả dụng rồi thoát")
    ap.add_argument("--save-snapshot", default="", help="lưu 1 frame ra file (path)")
    args = ap.parse_args()

    if args.scan:
        scan_cameras()
        return 0

    cap, w, h, declared_fps = open_camera(
        args.source, args.width, args.height, args.fps
    )
    if cap is None:
        print(f"❌ KHÔNG mở được camera: {args.source}")
        print("Thử: python scripts/test_webcam.py --scan")
        return 1

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"✅ Camera mở thành công: {args.source}")
    print(f"   Resolution thực tế: {w}x{h}")
    print(f"   FPS theo driver  : {declared_fps:.1f}")
    print(f"   Codec (FOURCC)   : {fourcc_str}")
    print(f"   Đo FPS thực trong {args.duration}s...")
    print("   (nhấn Q để thoát sớm khi headless=False)")
    print()

    t0 = time.time()
    deadline = t0 + args.duration
    frame_count = 0
    drops = 0
    snapshot_saved = False

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            drops += 1
            time.sleep(0.01)
            continue
        frame_count += 1

        if args.save_snapshot and not snapshot_saved:
            try:
                Path(args.save_snapshot).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(args.save_snapshot, frame)
                snapshot_saved = True
                print(f"   📸 Đã lưu snapshot: {args.save_snapshot}")
            except Exception as e:
                print(f"   ❌ Không lưu được snapshot: {e}")

        if not args.headless:
            elapsed = time.time() - t0
            cur_fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                frame, f"FPS: {cur_fps:.1f}  frames={frame_count}  drops={drops}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow("Webcam Test (Q=thoat)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0

    print()
    print("===== KẾT QUẢ =====")
    print(f"  Frames đọc được : {frame_count}")
    print(f"  Frames bị drop  : {drops}")
    print(f"  Thời gian       : {elapsed:.2f}s")
    print(f"  FPS trung bình  : {avg_fps:.1f}")

    if avg_fps < args.fps * 0.7:
        print(f"  ⚠️  FPS thấp hơn target ({args.fps}). Có thể do:")
        print("     - Camera không hỗ trợ resolution+FPS này (giảm xuống 640x480)")
        print("     - USB 2.0 không đủ băng thông cho MJPG 720p")
        print("     - Driver chưa tối ưu — thử thay cáp/cổng USB khác")
    elif drops > frame_count * 0.05:
        print(f"  ⚠️  Tỷ lệ drop cao ({drops}/{frame_count}) — kiểm tra cáp")
    else:
        print("  ✅ Camera ổn, sẵn sàng dùng cho main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
