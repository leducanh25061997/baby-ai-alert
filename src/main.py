import sys
# Đảm bảo print() ra ngay không bị buffer khi chạy dưới systemd.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# === Fail-fast environment guard ===
# Mediapipe 0.10.x compile chống NumPy 1.x → từ chối start nếu numpy>=2.
try:
    import numpy as np
except ImportError:
    sys.exit("❌ numpy chưa cài. Chạy: pip install -r requirements.txt")

_NP_MAJOR = int(np.__version__.split('.')[0])
if _NP_MAJOR >= 2:
    sys.exit(
        f"\n❌ NumPy {np.__version__} không tương thích với mediapipe 0.10.x.\n"
        f"   Chạy 1 lệnh fix cả 2 lib cùng lúc:\n"
        f"     python -m pip install 'numpy<2' 'opencv-python<4.11' --force-reinstall\n"
        f"   Hoặc dùng helper:  bash scripts/fix_env.sh  (Linux/macOS)\n"
        f"                       .\\scripts\\fix_env.ps1   (Windows)\n"
    )

# Check opencv version VIA METADATA (không import cv2) — opencv >=4.11 compile
# chống NumPy 2.x, mismatch với numpy<2 → silent corruption hoặc crash.
# Phải check CẢ 4 variant vì share cv2 namespace.
try:
    from importlib.metadata import version as _pkg_ver, PackageNotFoundError
    _cv_pkg_names = [
        'opencv-python', 'opencv-contrib-python',
        'opencv-python-headless', 'opencv-contrib-python-headless',
    ]
    _bad_cv = []
    _installed_cv = []
    for _name in _cv_pkg_names:
        try:
            _cv_ver = _pkg_ver(_name)
            _installed_cv.append((_name, _cv_ver))
            _cv_parts = _cv_ver.split('.')
            _cv_major, _cv_minor = int(_cv_parts[0]), int(_cv_parts[1])
            if _cv_major > 4 or (_cv_major == 4 and _cv_minor >= 11):
                _bad_cv.append(f"{_name}=={_cv_ver}")
        except PackageNotFoundError:
            continue
    if _bad_cv:
        sys.exit(
            f"\n❌ OpenCV variant compile chống NumPy 2.x, không khớp với "
            f"numpy<2 đang cài:\n"
            f"     {', '.join(_bad_cv)}\n\n"
            f"   Fix triệt để (uninstall tất cả variant rồi cài đúng):\n"
            f"     bash scripts/fix_env.sh           (Linux/macOS)\n"
            f"     .\\scripts\\fix_env.ps1            (Windows)\n"
        )
    if len(_installed_cv) > 1:
        # Cài nhiều variant cùng lúc → share cv2 namespace → bug khó debug
        sys.exit(
            f"\n❌ Phát hiện nhiều opencv variant cài cùng lúc (xung đột cv2 "
            f"namespace):\n"
            + "\n".join(f"     {n}=={v}" for n, v in _installed_cv) +
            f"\n\n   Fix: chạy scripts/fix_env.sh (uninstall hết rồi cài lại "
            f"chỉ opencv-python).\n"
        )
except SystemExit:
    raise
except Exception:
    pass  # không cài qua pip → bỏ qua check

import cv2
import mediapipe as mp
import time, asyncio, threading, io, json, os, signal
from datetime import datetime
from pathlib import Path
from telegram import Bot

from state_machine import (
    OcclusionStateMachine,
    STATE_ALERT, STATE_SAFE, STATE_NO_FACE, STATE_CALIBRATING,
    TRIGGER_FACE_LOST,
)
from occlusion_detector import (
    OcclusionDetector, CheckResult,
    NOSE_TIP, MOUTH_CENTER, PATCH_SIZE,
)

# ===================== CONFIG =====================
TELEGRAM_TOKEN = os.environ.get(
    "TELEGRAM_TOKEN",
    "8684958351:AAHE0XavsY_DgzEevmtcMUHB3_N3QwuNYIk",
)
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7316578932")

OCCLUSION_THRESHOLD_SEC = int(os.environ.get("OCCLUSION_THRESHOLD_SEC", "15"))
COOLDOWN_SEC            = int(os.environ.get("COOLDOWN_SEC",            "60"))

CALIBRATION_SEC         = int(os.environ.get("CALIBRATION_SEC",         "5"))
CONFIRM_FRAMES          = int(os.environ.get("CONFIRM_FRAMES",          "10"))
SMOOTHER_MAX_MISS       = int(os.environ.get("SMOOTHER_MAX_MISS",       "3"))
MIN_QUALITY_WARN        = float(os.environ.get("MIN_QUALITY_WARN",      "0.5"))

FACE_LOST_GRACE_SEC     = 1.5
FACE_GONE_RESET_SEC     = OCCLUSION_THRESHOLD_SEC + 10

# Auto re-calibrate sau khi safe liên tục bao lâu (giúp adapt môi trường đổi
# dần). 0 = tắt. Default 30 phút.
AUTO_RECAL_AFTER_SEC    = int(os.environ.get("AUTO_RECAL_AFTER_SEC", "1800"))

# YOLO settings (optional)
YOLO_PERSON_CONF        = 0.35
YOLO_RUN_EVERY_N_FRAMES = int(os.environ.get("YOLO_EVERY", "0")) or None
YOLO_CACHE_TTL_SEC      = 1.0
YOLO_DEVICE             = os.environ.get("YOLO_DEVICE", "auto")

# Camera
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "0")
CAMERA_WIDTH  = int(os.environ.get("CAMERA_WIDTH",  "1280"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT",  "720"))
CAMERA_FPS    = int(os.environ.get("CAMERA_FPS",     "30"))

HEADLESS = os.environ.get("HEADLESS", "0").lower() in ("1", "true", "yes")

# Detection mode — chọn cách phát hiện che mũi/miệng:
#   "multi_signal" (default): 4-signal voting (histogram + skin + edge + lap_var).
#                  Có thể bị defeat bởi khăn có nếp / patterned blanket.
#   "strict":      bỏ multi-signal, dùng face_mesh confidence cao (0.85). Khi
#                  mặt bị che một phần → confidence drop → face_lost trigger
#                  alert sau 15s. SAFETY-FIRST: nhiều false positive (mặt nghiêng,
#                  ánh sáng kém) NHƯNG không miss khi thật sự bị che.
DETECTION_MODE = os.environ.get("DETECTION_MODE", "multi_signal").lower()
if DETECTION_MODE not in ("multi_signal", "strict"):
    print(f"⚠️  DETECTION_MODE={DETECTION_MODE} không hợp lệ, fallback multi_signal")
    DETECTION_MODE = "multi_signal"

# Mediapipe confidence — strict mode dùng giá trị cao để dễ trigger face_lost
MP_DETECTION_CONFIDENCE = (
    0.85 if DETECTION_MODE == "strict"
    else float(os.environ.get("MP_DETECTION_CONFIDENCE", "0.6"))
)
MP_TRACKING_CONFIDENCE = (
    0.85 if DETECTION_MODE == "strict"
    else float(os.environ.get("MP_TRACKING_CONFIDENCE", "0.6"))
)

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
EVENTS_DIR      = PROJECT_ROOT / "events"
EVENTS_DIR.mkdir(exist_ok=True)
YOLO_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"
# ==================================================

mp_face_mesh = mp.solutions.face_mesh


class SmoothingBuffer:
    """Temporal smoothing với tolerance — cho phép 1-2 frame miss vì:
       - Mediapipe có thể jump landmark đôi khi
       - Trẻ cử động làm patch nhảy 1 frame
    Yêu cầu: ≥ (confirm - max_miss) trong window confirm gần nhất để CONFIRM.
    """

    def __init__(self, confirm=CONFIRM_FRAMES, clear=8, max_miss=SMOOTHER_MAX_MISS):
        self.buf      = []
        self.confirm  = confirm
        self.clear    = clear
        self.max_miss = max_miss   # cho phép 2 false trong 15 → vẫn confirm
        self.state    = False

    def update(self, val: bool) -> bool:
        self.buf.append(val)
        if len(self.buf) > max(self.confirm, self.clear):
            self.buf.pop(0)
        # Confirm: ≥ (confirm - max_miss) trong window phải True
        if len(self.buf) >= self.confirm:
            recent_true = sum(self.buf[-self.confirm:])
            if recent_true >= self.confirm - self.max_miss:
                self.state = True
        # Clear: ≤ max_miss True trong window clear → bỏ alert
        if len(self.buf) >= self.clear:
            recent_true = sum(self.buf[-self.clear:])
            if recent_true <= self.max_miss:
                self.state = False
        return self.state

    def reset(self):
        self.buf   = []
        self.state = False


def _resolve_yolo_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    if requested == "cuda":
        print("⚠️  YOLO_DEVICE=cuda nhưng torch.cuda không available → fallback cpu")
    return "cpu"


class PersonDetector:
    """YOLO wrapper — chỉ dùng khi mất mặt để phân biệt 'rời khung' vs 'bị phủ'."""

    def __init__(self, model_path: Path):
        self.model        = None
        self.device       = "cpu"
        self.run_every    = 5
        self._last_call_t = 0.0
        self._last_result = None
        self._frame_count = 0

        try:
            from ultralytics import YOLO
        except Exception as e:
            print(f"⚠️  Không có ultralytics ({e}); YOLO bị tắt.")
            return
        if not model_path.exists():
            print(f"⚠️  Không thấy model {model_path.name}; YOLO bị tắt.")
            return
        try:
            self.device = _resolve_yolo_device(YOLO_DEVICE)
            self.model  = YOLO(str(model_path))
            self.model.to(self.device)
            if YOLO_RUN_EVERY_N_FRAMES is not None:
                self.run_every = YOLO_RUN_EVERY_N_FRAMES
            else:
                self.run_every = 2 if self.device.startswith("cuda") else 5
            print(f"✅ YOLO sẵn sàng: {model_path.name} | "
                  f"device={self.device} | run_every={self.run_every}")
        except Exception as e:
            print(f"⚠️  Lỗi load YOLO: {e}; YOLO bị tắt.")
            self.model = None

    def has_person(self, frame):
        if self.model is None:
            return None
        now = time.monotonic()
        self._frame_count += 1

        cache_fresh = (self._last_result is not None
                       and now - self._last_call_t < YOLO_CACHE_TTL_SEC)
        if (self._frame_count % self.run_every != 0) and cache_fresh:
            return self._last_result

        try:
            results = self.model(
                frame, classes=[0], conf=YOLO_PERSON_CONF,
                device=self.device, verbose=False,
            )
            found = False
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    found = True
                    break
            self._last_call_t = now
            self._last_result = found
            return found
        except Exception as e:
            print(f"❌ Lỗi YOLO: {e}")
            return None


class BabyMonitorV5:
    def __init__(self):
        self.bot             = Bot(token=TELEGRAM_TOKEN)
        self.detector        = OcclusionDetector()
        self.smoother        = SmoothingBuffer()
        self.person_detector = PersonDetector(YOLO_MODEL_PATH)
        self.fsm             = OcclusionStateMachine(
            threshold_sec  = OCCLUSION_THRESHOLD_SEC,
            grace_sec      = FACE_LOST_GRACE_SEC,
            gone_reset_sec = FACE_GONE_RESET_SEC,
        )
        self.last_alert_time = 0.0
        self._prev_in_alert  = False
        # Trigger để recalibrate (set bởi 'R' hotkey hoặc SIGUSR1).
        self._recal_request  = threading.Event()
        # Theo dõi để auto-recalibrate khi safe lâu.
        self._safe_streak_start = None

        self._shutdown = threading.Event()
        try:
            signal.signal(signal.SIGTERM, lambda *_: self._shutdown.set())
            signal.signal(signal.SIGINT,  lambda *_: self._shutdown.set())
            # SIGUSR1 chỉ có trên Unix → trigger recalibrate qua `kill -USR1 <pid>`
            if hasattr(signal, "SIGUSR1"):
                signal.signal(signal.SIGUSR1,
                              lambda *_: self._recal_request.set())
        except ValueError:
            pass

    # ---------- Event persistence ----------
    def _save_event(self, frame, status, result, trigger):
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_iso  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            jpg_path  = EVENTS_DIR / f"possible_suffocation_risk_{ts_file}.jpg"
            json_path = EVENTS_DIR / f"possible_suffocation_risk_{ts_file}.json"
            cv2.imwrite(str(jpg_path), frame)
            extra = {
                "status": status,
                "alert_seconds": OCCLUSION_THRESHOLD_SEC,
                "trigger": trigger,
                "calibration_quality": self.detector.calibration_quality,
            }
            if result is not None:
                # Giữ tên 'upper'/'lower' cho khớp schema cũ
                extra.update({
                    "upper_score": float(result.nose_hist_corr),
                    "lower_score": float(result.mouth_hist_corr),
                    "nose_skin_ratio": float(result.nose_skin_ratio),
                    "mouth_skin_ratio": float(result.mouth_skin_ratio),
                    "nose_edge_density": float(result.nose_edge_density),
                    "mouth_edge_density": float(result.mouth_edge_density),
                    "nose_lap_var": float(result.nose_lap_var),
                    "mouth_lap_var": float(result.mouth_lap_var),
                    "nose_votes": int(result.nose_votes_for_occluded),
                    "mouth_votes": int(result.mouth_votes_for_occluded),
                })
            payload = {
                "event_type": "possible_suffocation_risk",
                "time": ts_iso,
                "extra": extra,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"💾 Đã lưu event: {jpg_path.name}")
        except Exception as e:
            print(f"❌ Lỗi lưu event: {e}")

    # ---------- Telegram alert ----------
    async def send_alert(self, frame, elapsed, result, trigger):
        now = time.monotonic()
        if now - self.last_alert_time < COOLDOWN_SEC:
            return
        self.last_alert_time = now

        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_io = io.BytesIO(buf.tobytes())
        img_io.name = 'alert.jpg'

        if trigger == TRIGGER_FACE_LOST:
            reason_block = "🫥 Mất hoàn toàn khuôn mặt — nghi bị phủ kín bởi vật lạ"
        elif result is not None:
            reason_block = (
                f"👃 Mũi vote: `{result.nose_votes_for_occluded}/4`  "
                f"hist=`{result.nose_hist_corr:.2f}` "
                f"skin=`{result.nose_skin_ratio:.2f}` "
                f"edge=`{result.nose_edge_density:.3f}` "
                f"lap=`{result.nose_lap_var:.0f}`\n"
                f"👄 Miệng vote: `{result.mouth_votes_for_occluded}/4`  "
                f"hist=`{result.mouth_hist_corr:.2f}` "
                f"skin=`{result.mouth_skin_ratio:.2f}` "
                f"edge=`{result.mouth_edge_density:.3f}` "
                f"lap=`{result.mouth_lap_var:.0f}`"
            )
        else:
            reason_block = "Phát hiện che mũi/miệng"

        caption = (
            f"🚨 *CẢNH BÁO NGẠT THỞ!*\n\n"
            f"⏰ Thời gian: `{ts}`\n"
            f"⏱ Bị che: `{elapsed:.1f} giây`\n\n"
            f"{reason_block}\n\n"
            f"⚠️ Mũi/miệng bị che quá {OCCLUSION_THRESHOLD_SEC}s!\n"
            f"👉 *Kiểm tra trẻ ngay lập tức!*"
        )

        try:
            await self.bot.send_photo(
                chat_id=CHAT_ID, photo=img_io,
                caption=caption, parse_mode='Markdown'
            )
            print(f"[{ts}] ✅ Đã gửi cảnh báo Telegram!")
        except Exception as e:
            print(f"❌ Lỗi Telegram: {e}")

    def _dispatch_alert(self, frame, elapsed, result, trigger):
        status = ("FULLY_COVERED" if trigger == TRIGGER_FACE_LOST
                  else "POSSIBLE_OCCLUSION")
        snap = frame.copy()
        self._save_event(snap, status, result, trigger)
        threading.Thread(
            target=lambda: asyncio.run(
                self.send_alert(snap, elapsed, result, trigger)
            ), daemon=True
        ).start()

    def _request_recalibrate(self, reason: str):
        print(f"🔄 Yêu cầu recalibrate: {reason}")
        self._recal_request.set()

    def _do_recalibrate(self):
        """Reset toàn bộ state để vào lại CALIBRATING."""
        self.detector.reset()
        self.smoother.reset()
        self.fsm._reset()   # noqa
        self.fsm.last_face_seen = None
        self._safe_streak_start = None
        self._recal_request.clear()
        print("🔄 Đã reset — sẽ calibrate lại khi thấy mặt.\n")

    # ---------- UI ----------
    def draw_ui(self, frame, landmarks, state, elapsed, face_present,
                result, calib_remaining, w, h, trigger="", person_seen=None):

        # Draw landmark patches
        if landmarks and face_present and self.detector.is_ready:
            nose_lm  = landmarks[NOSE_TIP]
            mouth_lm = landmarks[MOUTH_CENTER]
            # Màu patch: đỏ nếu vote ≥2, vàng nếu vote=1, xanh nếu vote=0
            def vote_color(v):
                return (0,0,255) if v >= 2 else ((0,200,255) if v == 1 else (0,255,0))
            n_color = vote_color(result.nose_votes_for_occluded  if result else 0)
            m_color = vote_color(result.mouth_votes_for_occluded if result else 0)
            half = PATCH_SIZE // 2
            nx = int(nose_lm.x * w);  ny = int(nose_lm.y * h)
            mx = int(mouth_lm.x * w); my = int(mouth_lm.y * h)
            cv2.rectangle(frame, (nx-half, ny-half), (nx+half, ny+half), n_color, 2)
            cv2.rectangle(frame, (mx-half, my-half), (mx+half, my+half), m_color, 2)
            if result:
                cv2.putText(frame, f"N:{result.nose_votes_for_occluded}/4",
                            (nx-half, ny-half-6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, n_color, 1)
                cv2.putText(frame, f"M:{result.mouth_votes_for_occluded}/4",
                            (mx-half, my-half-6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, m_color, 1)

        person_label = (
            "YOLO:nguoi" if person_seen is True
            else "YOLO:vang" if person_seen is False
            else "YOLO:off"
        )
        if result is not None:
            debug = [
                f"NOSE  hist={result.nose_hist_corr:.2f}  skin={result.nose_skin_ratio:.2f}  edge={result.nose_edge_density:.3f}  lap={result.nose_lap_var:.0f}  V={result.nose_votes_for_occluded}/4",
                f"MOUTH hist={result.mouth_hist_corr:.2f}  skin={result.mouth_skin_ratio:.2f}  edge={result.mouth_edge_density:.3f}  lap={result.mouth_lap_var:.0f}  V={result.mouth_votes_for_occluded}/4",
                f"Quality:{self.detector.calibration_quality:.2f}  {person_label}  (R=recal)",
            ]
        else:
            debug = [
                f"Detector chua ready  {person_label}",
                "",
                "(R=recal)",
            ]
        for i, line in enumerate(debug):
            cv2.putText(frame, line, (10, 50 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)

        if state == STATE_ALERT:
            cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,220), 6)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-65), (w, h), (15,15,15), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        if state == STATE_CALIBRATING:
            msg   = f"Calibration... Giu nguyen mat tre {calib_remaining:.1f}s"
            color = (0, 200, 255)
            progress = 1.0 - (calib_remaining / CALIBRATION_SEC)
            cv2.rectangle(frame, (0, h-5), (int(w*progress), h), (0,200,255), -1)
        elif state == STATE_NO_FACE:
            msg   = "Khong phat hien khuon mat"
            color = (100, 100, 100)
        elif state == STATE_ALERT:
            rem = max(0, OCCLUSION_THRESHOLD_SEC - elapsed)
            if trigger == TRIGGER_FACE_LOST:
                msg = f"MAT MAT - NGHI BI PHU KIN {elapsed:.1f}s | Con lai: {rem:.1f}s"
            else:
                msg = f"MUI/MIENG BI CHE {elapsed:.1f}s | Con lai: {rem:.1f}s"
            color = (0, 80, 255)
            cv2.putText(frame, ">>> NGUY CO NGAT THO <<<",
                        (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,60,255), 2)
        else:
            quality_warn = ""
            if self.detector.is_ready and self.detector.calibration_quality < MIN_QUALITY_WARN:
                quality_warn = f"  [Q={self.detector.calibration_quality:.2f} thap — nhan R de recalib]"
            msg   = "Binh thuong - Mui/mieng dang nhin thay" + quality_warn
            color = (0, 220, 0)

        cv2.putText(frame, msg, (10, h-35 if state == STATE_ALERT else h-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        dot_color = {
            STATE_CALIBRATING: (0,200,255),
            STATE_SAFE:        (0,255,0),
            STATE_ALERT:       (0,0,255),
            STATE_NO_FACE:     (128,128,128),
        }.get(state, (128,128,128))
        cv2.circle(frame, (w-20, 20), 10, dot_color, -1)
        return frame

    # ---------- Main loop ----------
    def run(self):
        src = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"❌ Không mở được camera: {src}")
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS,    0)

        actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print("🟢 Baby Monitor V5")
        print(f"   Detection mode   : {DETECTION_MODE.upper()}")
        if DETECTION_MODE == "strict":
            print(f"   → Strict mode: KHÔNG dùng multi-signal voting.")
            print(f"   → MediaPipe confidence cao ({MP_DETECTION_CONFIDENCE}) — face_lost trigger handle alert")
        print(f"   Camera           : {src} → {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
        print(f"   Calibration      : {CALIBRATION_SEC}s")
        print(f"   Cảnh báo sau     : {OCCLUSION_THRESHOLD_SEC}s")
        print(f"   Auto-recal       : sau {AUTO_RECAL_AFTER_SEC}s safe liên tục"
              if AUTO_RECAL_AFTER_SEC > 0 else "   Auto-recal       : tắt")
        print(f"   Events lưu tại   : {EVENTS_DIR}")
        print(f"   Headless         : {HEADLESS}")
        print(f"   Recal manual     : nhấn R (GUI) hoặc `kill -USR1 <pid>` (headless)")
        print("\n⏳ Đang load MediaPipe (lần đầu có thể mất 5-10s)...")

        calib_start = None
        # Strict mode không cần calibration (không dùng multi-signal)
        calib_done  = (DETECTION_MODE == "strict")
        if calib_done:
            print("ℹ️  STRICT mode — bỏ qua calibration phase.")
        fail_count  = 0
        MAX_FAILS   = 60

        try:
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=MP_DETECTION_CONFIDENCE,
                min_tracking_confidence=MP_TRACKING_CONFIDENCE,
            ) as face_mesh:
                print("✅ MediaPipe sẵn sàng. Đang chờ mặt trẻ để calibrate...\n")

                while not self._shutdown.is_set() and cap.isOpened():
                    # === Recalibrate trigger ===
                    if self._recal_request.is_set():
                        self._do_recalibrate()
                        calib_start = None
                        calib_done  = False

                    ret, frame = cap.read()
                    if not ret:
                        fail_count += 1
                        if fail_count >= MAX_FAILS:
                            print(f"❌ Mất camera {fail_count} frame → thoát")
                            break
                        time.sleep(0.05)
                        continue
                    fail_count = 0

                    h, w  = frame.shape[:2]
                    now   = time.monotonic()
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res   = face_mesh.process(rgb)
                    face_present = bool(res.multi_face_landmarks)

                    state           = STATE_NO_FACE
                    calib_remaining = 0.0
                    elapsed         = 0.0
                    trigger         = ""
                    person_seen     = None
                    check_result: CheckResult | None = None

                    if calib_done and not face_present:
                        person_seen = self.person_detector.has_person(frame)

                    if face_present and not calib_done:
                        # === Calibration ===
                        lms = res.multi_face_landmarks[0].landmark
                        if calib_start is None:
                            calib_start = now
                            print("✅ Phát hiện mặt! Đang calibrate baseline...")
                        calib_remaining = max(0, CALIBRATION_SEC - (now - calib_start))
                        self.detector.add_calibration_sample(frame, lms, w, h)
                        if now - calib_start >= CALIBRATION_SEC:
                            ok, msg = self.detector.finalize_calibration()
                            if ok:
                                calib_done = True
                                print(f"✅ Calibration xong! {msg}\n")
                                if self.detector.calibration_quality < MIN_QUALITY_WARN:
                                    print(f"⚠️  Quality={self.detector.calibration_quality:.2f} "
                                          f"hơi thấp. Cân nhắc recalibrate (nhấn R).")
                            else:
                                # Critical fail → reset, thử lại
                                print(f"❌ Calibration fail: {msg}")
                                print("   Sẽ thử lại sau 2 giây...")
                                self.detector.reset()
                                calib_start = None
                                time.sleep(2.0)
                        state = STATE_CALIBRATING

                    elif not face_present and not calib_done:
                        calib_start = None
                        state = STATE_NO_FACE

                    else:
                        # === Monitoring ===
                        occluded_by_detector = False
                        if face_present:
                            lms = res.multi_face_landmarks[0].landmark
                            if DETECTION_MODE == "strict":
                                # STRICT mode: bỏ multi-signal voting hoàn toàn.
                                # Chỉ trust face_present từ mediapipe (high confidence).
                                # Khi bị che → mediapipe drop face_present=False →
                                # state machine's face_lost path alert sau 15s.
                                check_result = None
                                occluded_by_detector = False
                            else:
                                check_result = self.detector.check(
                                    frame, lms, w, h,
                                    prev_in_alert=self._prev_in_alert,
                                )
                                if check_result is not None:
                                    occluded_by_detector = self.smoother.update(
                                        check_result.occluded
                                    )
                        else:
                            self.smoother.reset()

                        result_fsm = self.fsm.step(
                            now=now,
                            face_present=face_present,
                            occluded_by_histogram=occluded_by_detector,
                            person_in_frame=person_seen,
                        )
                        state   = result_fsm.state
                        elapsed = result_fsm.elapsed
                        trigger = result_fsm.trigger

                        if result_fsm.should_alert:
                            print(f"🚨 Đủ ngưỡng {OCCLUSION_THRESHOLD_SEC}s — "
                                  f"trigger={trigger} elapsed={elapsed:.1f}s")
                            self._dispatch_alert(frame, elapsed,
                                                 check_result, trigger)

                        # === Auto-recalibrate sau N giây safe liên tục ===
                        if AUTO_RECAL_AFTER_SEC > 0 and state == STATE_SAFE:
                            if self._safe_streak_start is None:
                                self._safe_streak_start = now
                            elif now - self._safe_streak_start > AUTO_RECAL_AFTER_SEC:
                                self._request_recalibrate(
                                    f"safe liên tục {AUTO_RECAL_AFTER_SEC}s — refresh baseline"
                                )
                        else:
                            self._safe_streak_start = None

                    # Cập nhật prev alert cho frame sau
                    self._prev_in_alert = (state == STATE_ALERT)

                    # === UI + hotkey 'R' ===
                    if not HEADLESS:
                        lms_draw = (res.multi_face_landmarks[0].landmark
                                    if face_present else None)
                        frame = self.draw_ui(
                            frame, lms_draw, state, elapsed, face_present,
                            check_result, calib_remaining, w, h,
                            trigger=trigger, person_seen=person_seen,
                        )
                        cv2.imshow('Baby Monitor V5 | Q=thoat R=recal', frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('r'):
                            self._request_recalibrate("user nhấn R")
        except KeyboardInterrupt:
            print("\n⏹  Nhận Ctrl+C, dừng...")
        except Exception as e:
            print(f"💥 Lỗi không xử lý được: {type(e).__name__}: {e}")
            raise
        finally:
            cap.release()
            if not HEADLESS:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            print("🔴 Đã dừng")


if __name__ == "__main__":
    BabyMonitorV5().run()
