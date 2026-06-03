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
from scene_monitor import SceneMonitor
from alert_policy import (
    Watchdog, HeartbeatPolicy, CalibrationReminder, CalibrationConditionWarner,
)

# ===================== CONFIG =====================
# Defaults được tune sẵn cho Raspberry Pi 4 Model B (BCM2711, 4 nhân Cortex-A72,
# ARM64, CPU-only). Mỗi giá trị vẫn override được qua env, nhưng deploy production
# trên Pi 4 KHÔNG cần file .env nào cho phần hiệu năng — chạy thẳng src/main.py là
# ra đúng 640x480@15. File .env giờ chỉ cần để đặt TELEGRAM_TOKEN/CHAT_ID thật.
#
# NVIDIA/CUDA: KHÔNG hỗ trợ. Project chạy hoàn toàn CPU.
#   - MediaPipe FaceMesh: tối ưu sẵn cho ARM CPU.
#   - YOLOv8n: model nano (~6MB), CPU đủ nhanh khi YOLO_EVERY=10 trên Pi 4.
#   - Pi 4 KHÔNG có NPU. Muốn tăng tốc YOLO → gắn Google Coral USB TPU
#     (xem INSTALL_PI4.md §13). Không bắt buộc cho mục đích phát hiện che mũi/miệng.

TELEGRAM_TOKEN = os.environ.get(
    "TELEGRAM_TOKEN",
    "8684958351:AAHE0XavsY_DgzEevmtcMUHB3_N3QwuNYIk",
)
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7316578932")

OCCLUSION_THRESHOLD_SEC = int(os.environ.get("OCCLUSION_THRESHOLD_SEC", "15"))
# Spam control trong send_alert chỉ là defense-in-depth chống race condition
# (state_machine đã quản lý timing chính: re-alert mỗi OCCLUSION_THRESHOLD_SEC).
# Đặt mặc định 5s — đủ chặn double-fire frame-level mà không chặn re-alert
# chính đáng mỗi 15s khi vẫn còn bị che.
COOLDOWN_SEC            = int(os.environ.get("COOLDOWN_SEC",            "5"))

CALIBRATION_SEC         = int(os.environ.get("CALIBRATION_SEC",         "5"))
CONFIRM_FRAMES          = int(os.environ.get("CONFIRM_FRAMES",          "10"))
SMOOTHER_MAX_MISS       = int(os.environ.get("SMOOTHER_MAX_MISS",       "3"))
MIN_QUALITY_WARN        = float(os.environ.get("MIN_QUALITY_WARN",      "0.5"))

# Tần suất in heartbeat log tiếng Việt khi chạy headless (Pi 4 chạy không màn hình).
# Mỗi N giây in lại trạng thái hiện tại để vận hành biết hệ thống vẫn sống và
# đang ở trạng thái nào. State chuyển → luôn in ngay không phụ thuộc giá trị này.
LOG_INTERVAL_SEC        = float(os.environ.get("LOG_INTERVAL_SEC",      "5"))

FACE_LOST_GRACE_SEC     = 1.5
FACE_GONE_RESET_SEC     = OCCLUSION_THRESHOLD_SEC + 10

# Auto re-calibrate sau khi safe liên tục bao lâu (giúp adapt môi trường đổi
# dần). 0 = tắt. Default 30 phút.
AUTO_RECAL_AFTER_SEC    = int(os.environ.get("AUTO_RECAL_AFTER_SEC", "1800"))

# ---- Blur gate (chống false-positive do ảnh mờ) ----
# current_sharpness < BLUR_DROP_FRAC * baseline → coi cả khung đang mờ → bỏ phiếu
# edge/lap vòng đó. Bật mặc định (cải thiện thuần, không có nhược điểm rõ rệt).
BLUR_DROP_FRAC          = float(os.environ.get("BLUR_DROP_FRAC", "0.45"))

# ---- Watchdog / heartbeat (tự giám sát, không fail âm thầm) ----
# Cảnh báo suy giảm (camera đơ / quá tối / FPS thấp) khi kéo dài >= ngần này giây.
HEALTH_DEGRADE_SEC      = float(os.environ.get("HEALTH_DEGRADE_SEC", "20"))
HEALTH_MIN_FPS          = float(os.environ.get("HEALTH_MIN_FPS", "3.0"))
HEALTH_DARK_LUMA        = float(os.environ.get("HEALTH_DARK_LUMA", "25"))
# Heartbeat "vẫn đang canh" gửi Telegram mỗi ngần này giây. 0 = tắt (mặc định).
# Vd 21600 = mỗi 6 giờ.
HEARTBEAT_SEC           = int(os.environ.get("HEARTBEAT_SEC", "0"))

# ---- Thông báo khởi động / hiệu chỉnh ----
# Lúc khởi động gửi Telegram yêu cầu đưa mặt trẻ vào khung để hiệu chỉnh; nhắc
# lại mỗi CALIB_REMIND_SEC nếu mãi chưa thấy mặt; xác nhận khi hiệu chỉnh xong.
STARTUP_NOTIFY          = os.environ.get("STARTUP_NOTIFY", "1").lower() in ("1", "true", "yes")
CALIB_REMIND_SEC        = int(os.environ.get("CALIB_REMIND_SEC", "60"))
# Cổng chất lượng khi hiệu chỉnh: chỉ học baseline từ frame ĐẠT (đủ sáng + nét).
# Điều kiện kém kéo dài ≥ ngần này giây → gửi Telegram hướng dẫn (grace để bỏ
# qua mờ thoáng qua do autofocus). Ngưỡng tối dùng chung HEALTH_DARK_LUMA.
CALIB_COND_GRACE_SEC    = float(os.environ.get("CALIB_COND_GRACE_SEC", "4"))

# YOLO settings (CPU-only). Default run_every=10 frame để giảm tải trên Pi 4
# (4 nhân A72 yếu hơn → giãn YOLO; YOLO chỉ dùng khi mất mặt nên không hại độ nhạy).
YOLO_PERSON_CONF        = 0.35
YOLO_RUN_EVERY_N_FRAMES = int(os.environ.get("YOLO_EVERY", "10"))
YOLO_CACHE_TTL_SEC      = 1.0

# Camera defaults cho Logitech webcam qua USB 3.0 trên Raspberry Pi 4.
# 640x480@15 là cấu hình giữ được ≥6 FPS end-to-end trên Pi 4 (720p sẽ tụt còn
# ~4-6 FPS → ảnh mờ → false alert). Camera đặt xa có thể thử 800x600.
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "0")
CAMERA_WIDTH  = int(os.environ.get("CAMERA_WIDTH",  "640"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "480"))
CAMERA_FPS    = int(os.environ.get("CAMERA_FPS",    "15"))
# Autofocus: MẶC ĐỊNH BẬT (1). Tắt autofocus + đối tượng đổi khoảng cách → ảnh
# out-focus → vùng mũi/miệng mờ → edge/lap_var tụt về ~0 → detector tưởng "bị
# che" → FALSE ALERT. Chỉ nên tắt (CAMERA_AUTOFOCUS=0) khi camera + trẻ cố định
# tuyệt đối và bạn muốn tránh autofocus "hunting".
CAMERA_AUTOFOCUS = os.environ.get("CAMERA_AUTOFOCUS", "1").lower() in ("1", "true", "yes")

# Production trên Pi 4 không có HDMI → default headless=1.
# Dev local muốn xem live view: set HEADLESS=0 khi chạy.
HEADLESS = os.environ.get("HEADLESS", "1").lower() in ("1", "true", "yes")

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


class PersonDetector:
    """YOLO wrapper — chỉ dùng khi mất mặt để phân biệt 'rời khung' vs 'bị phủ'.
    CPU-only: project không hỗ trợ NVIDIA GPU; Pi 4 không có NPU/CUDA.
    """

    def __init__(self, model_path: Path):
        self.model        = None
        self.device       = "cpu"
        self.run_every    = YOLO_RUN_EVERY_N_FRAMES
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
            self.model = YOLO(str(model_path))
            self.model.to(self.device)
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
                device="cpu", verbose=False,
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
        self.detector        = OcclusionDetector()
        self.smoother        = SmoothingBuffer()
        self.person_detector = PersonDetector(YOLO_MODEL_PATH)
        self.scene           = SceneMonitor(
            blur_drop_frac = BLUR_DROP_FRAC,
        )
        self.fsm             = OcclusionStateMachine(
            threshold_sec  = OCCLUSION_THRESHOLD_SEC,
            grace_sec      = FACE_LOST_GRACE_SEC,
            gone_reset_sec = FACE_GONE_RESET_SEC,
        )
        self.last_alert_time = 0.0
        self._prev_in_alert  = False
        # --- Policy timing thuần (test ở tests/test_alert_policy.py) ---
        self.watchdog      = Watchdog(degrade_sec=HEALTH_DEGRADE_SEC)
        self.heartbeat     = HeartbeatPolicy(interval_sec=HEARTBEAT_SEC)
        self.calib_reminder = CalibrationReminder(
            enabled=STARTUP_NOTIFY, remind_sec=CALIB_REMIND_SEC)
        self.calib_cond_warner = CalibrationConditionWarner(
            enabled=STARTUP_NOTIFY, grace_sec=CALIB_COND_GRACE_SEC,
            remind_sec=CALIB_REMIND_SEC)
        self._frame_ts            = []      # timestamp các frame gần đây → FPS
        self._calib_done_notified = False   # đã gửi "bắt đầu giám sát" chưa (1 lần/phiên)
        # occlusion_start của event đã LƯU ẢNH gần nhất. Dùng để chỉ lưu
        # ảnh+json cho alert ĐẦU của mỗi event; các lần re-alert (mỗi 15s khi
        # vẫn còn bị che) chỉ gửi Telegram + log, KHÔNG ghi file trùng xuống
        # events/ (tránh phình đĩa khi che kéo dài / false-positive lâu).
        self._last_saved_occlusion_start = None
        # Trigger để recalibrate (set bởi 'R' hotkey hoặc SIGUSR1).
        self._recal_request  = threading.Event()
        # Theo dõi để auto-recalibrate khi safe lâu.
        self._safe_streak_start = None
        # State của lần log gần nhất + thời điểm heartbeat gần nhất.
        # Dùng để in chuyển trạng thái ngay + heartbeat định kỳ tiếng Việt
        # khi chạy headless trên Pi 4 (không có màn hình debug).
        self._last_logged_state = None
        self._last_heartbeat_t  = 0.0

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
            jpg_path  = EVENTS_DIR / f"face_covered_{ts_file}.jpg"
            json_path = EVENTS_DIR / f"face_covered_{ts_file}.json"
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
                "event_type": "face_covered",
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
        jpg_bytes = buf.tobytes()   # giữ bytes để tạo BytesIO mới cho mỗi lần retry

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
            f"🚨 *CẢNH BÁO CHE MŨI/MIỆNG!*\n\n"
            f"⏰ Thời gian: `{ts}`\n"
            f"⏱ Bị che: `{elapsed:.1f} giây`\n\n"
            f"{reason_block}\n\n"
            f"⚠️ Mũi/miệng bị che quá {OCCLUSION_THRESHOLD_SEC}s!\n"
            f"👉 *Kiểm tra trẻ ngay lập tức!*"
        )

        # Bot MỚI cho mỗi lần gửi: _dispatch_alert chạy mỗi alert trong 1
        # thread + event loop riêng (asyncio.run). python-telegram-bot v20+
        # bind httpx client nội bộ vào event loop ĐẦU TIÊN nó được dùng — tái
        # dùng 1 Bot chung ở loop sau (loop cũ đã đóng) sẽ treo vô hạn hoặc
        # crash, khiến chỉ alert đầu tiên tới được Telegram. Bot mới mỗi lần
        # → client bind đúng loop hiện tại → mọi re-alert đều gửi được.
        #
        # RETRY + BACKOFF: lúc autostart, alert đầu có thể fire khi WiFi/DNS
        # chưa sẵn (network-online chưa có) → send_photo timeout. Thay vì bỏ
        # luôn alert (gây "rất lâu mới tới Telegram"), thử lại với backoff tới
        # khi mạng lên. Tạo BytesIO mới mỗi lần (lần trước đã bị đọc cạn).
        delays = [0, 2, 5, 10, 20]   # tổng ~37s, đủ chờ WiFi/DNS lên sau boot
        for attempt, delay in enumerate(delays, start=1):
            if delay:
                await asyncio.sleep(delay)
            try:
                img_io = io.BytesIO(jpg_bytes)
                img_io.name = 'alert.jpg'
                async with Bot(token=TELEGRAM_TOKEN) as bot:
                    await bot.send_photo(
                        chat_id=CHAT_ID, photo=img_io,
                        caption=caption, parse_mode='Markdown',
                        read_timeout=20, write_timeout=20,
                        connect_timeout=20,
                    )
                suffix = f" (lần thử {attempt})" if attempt > 1 else ""
                print(f"[{ts}] ✅ Đã gửi cảnh báo Telegram!{suffix}")
                return
            except Exception as e:
                print(f"❌ Lỗi Telegram (thử {attempt}/{len(delays)}): {e}")
        print("❌ Gửi Telegram thất bại sau nhiều lần — ảnh vẫn đã lưu local "
              "trong events/ (xem lại khi có mạng).")

    def _warmup_telegram(self):
        """Hâm nóng kết nối Telegram ở thread nền lúc khởi động.

        Lúc autostart (systemd) máy vừa boot, WiFi/DNS có thể chưa sẵn. Thread
        này gọi getMe lặp lại tới khi được → vừa xác thực token, vừa làm ấm
        DNS/TLS để alert ĐẦU tiên sau đó gửi tức thì thay vì phải chờ mạng.
        Không chặn main loop (detection vẫn chạy bình thường)."""
        def _run():
            delay = 3
            for attempt in range(1, 13):   # ~ vài phút với backoff
                if self._shutdown.is_set():
                    return
                try:
                    async def _check():
                        async with Bot(token=TELEGRAM_TOKEN) as bot:
                            return (await bot.get_me()).username
                    name = asyncio.run(_check())
                    print(f"✅ Telegram sẵn sàng (bot @{name}) — kết nối đã hâm nóng.")
                    return
                except Exception as e:
                    print(f"⏳ Chờ mạng/Telegram (thử {attempt}): {type(e).__name__}")
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
            print("⚠️ Warm-up Telegram chưa xong sau nhiều lần — alert sẽ tự retry khi fire.")
        threading.Thread(target=_run, daemon=True).start()

    def _notify_text(self, text: str):
        """Gửi 1 tin nhắn TEXT Telegram ở thread nền (có retry) — dùng cho cảnh
        báo watchdog/heartbeat. KHÔNG lưu event (không phải sự kiện che mũi/miệng)."""
        def _run():
            for attempt, delay in enumerate([0, 3, 8], start=1):
                if delay:
                    time.sleep(delay)
                try:
                    async def _send():
                        async with Bot(token=TELEGRAM_TOKEN) as bot:
                            await bot.send_message(
                                chat_id=CHAT_ID, text=text, parse_mode='Markdown',
                                read_timeout=15, write_timeout=15, connect_timeout=15,
                            )
                    asyncio.run(_send())
                    return
                except Exception as e:
                    print(f"❌ Lỗi gửi text Telegram (thử {attempt}): {type(e).__name__}")
        threading.Thread(target=_run, daemon=True).start()

    def _dispatch_alert(self, frame, elapsed, result, trigger, save_image=True):
        status = ("FULLY_COVERED" if trigger == TRIGGER_FACE_LOST
                  else "POSSIBLE_OCCLUSION")
        snap = frame.copy()
        # Chỉ lưu ảnh+json cho alert đầu của event; re-alert chỉ gửi Telegram.
        if save_image:
            self._save_event(snap, status, result, trigger)
        threading.Thread(
            target=lambda: asyncio.run(
                self.send_alert(snap, elapsed, result, trigger)
            ), daemon=True
        ).start()

    # ---------- Watchdog (tự giám sát, không fail âm thầm) ----------
    def _update_fps(self, now):
        self._frame_ts.append(now)
        if len(self._frame_ts) > 30:
            self._frame_ts.pop(0)
        if len(self._frame_ts) >= 5:
            span = self._frame_ts[-1] - self._frame_ts[0]
            if span > 0:
                return (len(self._frame_ts) - 1) / span
        return None

    def _update_health(self, now, fps):
        """Gom các sự cố đang gặp → đưa Watchdog quyết định cảnh báo/khôi phục."""
        issues = {}
        if self.scene.is_frozen_frame:
            issues['camera_frozen'] = "📷 Camera bị ĐƠ (frame không đổi) — kiểm tra cáp/USB."
        if self.scene.luma < HEALTH_DARK_LUMA:
            issues['too_dark'] = "🌑 Phòng QUÁ TỐI — không giám sát tin cậy được (cần đèn/IR)."
        if fps is not None and fps < HEALTH_MIN_FPS:
            issues['low_fps'] = f"🐌 FPS quá thấp ({fps:.1f}) — hệ thống quá tải/nóng (kiểm tra nhiệt độ)."

        for ev in self.watchdog.step(now, issues):
            if ev.kind == 'alert':
                self._notify_text(f"⚠️ *GIÁM SÁT SUY GIẢM*\n{ev.message}")
                print(f"⚠️ Watchdog: '{ev.name}' kéo dài >{HEALTH_DEGRADE_SEC:.0f}s → đã cảnh báo Telegram.")
            else:
                self._notify_text(f"✅ Đã khôi phục — `{ev.name}` hết, giám sát trở lại bình thường.")
                print(f"✅ Watchdog: '{ev.name}' đã khôi phục.")

    def _maybe_heartbeat(self, now, state):
        if self.heartbeat.step(now):
            self._notify_text(f"💓 Baby Monitor vẫn đang canh — trạng thái: `{state}`.")

    def _maybe_remind_calibration(self, now, calib_done, face_present):
        """Nhắc đưa mặt vào khung khi mãi chưa hiệu chỉnh được (theo CalibrationReminder)."""
        if self.calib_reminder.step(now, calib_done, face_present):
            self._notify_text(
                "⚠️ *Chưa hiệu chỉnh được* — vẫn chưa thấy mặt trẻ trong khung.\n"
                "👉 Kiểm tra hướng camera / đưa mặt trẻ vào khung camera."
            )

    def _maybe_warn_calib_conditions(self, now, too_dark, too_blurry):
        """Đang thấy mặt để hiệu chỉnh nhưng điều kiện kém (tối/mờ) khiến không
        gom đủ mẫu tốt → Telegram hướng dẫn người dùng khắc phục (headless)."""
        conditions_bad = too_dark or too_blurry
        if not self.calib_cond_warner.step(now, calibrating=True,
                                           conditions_bad=conditions_bad):
            return
        if too_dark and too_blurry:
            detail, fix = ("phòng quá tối *và* hình bị mờ",
                           "bật thêm đèn và chỉnh lại tiêu cự/khoảng cách camera")
        elif too_dark:
            detail, fix = ("phòng quá tối",
                           "bật thêm đèn cho phòng")
        else:
            detail, fix = ("hình bị mờ (camera chưa nét)",
                           "chỉnh lại tiêu cự/khoảng cách camera và giữ máy cố định")
        self._notify_text(
            f"⚠️ *Chưa hiệu chỉnh được — {detail}.*\n"
            f"👉 Vui lòng {fix}, rồi giữ mặt trẻ trong khung ~5 giây.\n"
            f"   (Hệ thống đang chờ điều kiện tốt để học chuẩn — sẽ báo khi xong.)"
        )

    def _notify_calibrated(self):
        """Gửi xác nhận 'đã bắt đầu giám sát' — chỉ 1 lần/phiên chạy."""
        if not STARTUP_NOTIFY or self._calib_done_notified:
            return
        self._calib_done_notified = True
        q = self.detector.calibration_quality
        msg = (f"✅ *Đã hiệu chỉnh xong — BẮT ĐẦU GIÁM SÁT.*\n"
               f"   Chất lượng: `{q:.2f}`")
        if q < MIN_QUALITY_WARN:
            msg += ("\n⚠️ Chất lượng hơi thấp — nên tăng sáng / chỉnh lại vị trí "
                    "camera; hệ thống sẽ tự hiệu chỉnh lại khi ổn định.")
        self._notify_text(msg)

    def _log_status(self, now, state, elapsed, trigger, check_result,
                    person_seen, calib_remaining, face_present):
        """In log tiếng Việt cho trạng thái hiện tại.

        - State chuyển → in ngay (transition log).
        - State không chuyển → in heartbeat mỗi LOG_INTERVAL_SEC giây.
        Mục đích: vận hành trên Pi 4 headless biết hệ thống đang
        bị che / không bị che / mất mặt mà không cần GUI.
        """
        state_changed = state != self._last_logged_state
        time_for_heartbeat = (now - self._last_heartbeat_t) >= LOG_INTERVAL_SEC
        if not state_changed and not time_for_heartbeat:
            return

        ts = datetime.now().strftime("%H:%M:%S")
        prefix = "🔁" if state_changed else "  "

        if state == STATE_CALIBRATING:
            line = (f"{prefix} [{ts}] 🟡 ĐANG HIỆU CHỈNH baseline — "
                    f"giữ nguyên mặt trẻ, còn {calib_remaining:.1f}s")

        elif state == STATE_NO_FACE:
            yolo_note = ""
            if person_seen is True:
                yolo_note = " | YOLO: vẫn thấy người trong khung"
            elif person_seen is False:
                yolo_note = " | YOLO: không có người (trẻ rời khung)"
            line = f"{prefix} [{ts}] ⚪ KHÔNG THẤY MẶT TRẺ{yolo_note}"

        elif state == STATE_SAFE:
            extra = ""
            if check_result is not None:
                nv = check_result.nose_votes_for_occluded
                mv = check_result.mouth_votes_for_occluded
                if nv >= 1 or mv >= 1:
                    extra = (f" | ⚠ dấu hiệu nhẹ: mũi {nv}/4, miệng {mv}/4 "
                             f"(chưa đủ để cảnh báo)")
            line = (f"{prefix} [{ts}] 🟢 BÌNH THƯỜNG — Mũi/miệng nhìn rõ, "
                    f"KHÔNG bị che{extra}")

        elif state == STATE_ALERT:
            # Tính countdown đúng với reality của state_machine:
            #   - Chưa đủ ngưỡng (elapsed < threshold) → countdown tới alert đầu
            #   - Đã alert lần đầu → countdown tới lần re-alert tiếp theo
            #     (mỗi threshold_sec lặp 1 lần khi còn bị che)
            last_alert_at = self.fsm.last_alert_at
            if last_alert_at is None:
                rem = max(0.0, OCCLUSION_THRESHOLD_SEC - elapsed)
                countdown_txt = f"còn {rem:.1f}s nữa sẽ gửi cảnh báo Telegram"
            else:
                next_rem = max(
                    0.0,
                    OCCLUSION_THRESHOLD_SEC - (now - last_alert_at),
                )
                countdown_txt = (
                    f"đã gửi cảnh báo, gửi lại sau {next_rem:.1f}s "
                    f"nếu vẫn còn bị che"
                )
            if trigger == TRIGGER_FACE_LOST:
                line = (f"{prefix} [{ts}] 🔴 MẤT MẶT — NGHI BỊ PHỦ KÍN "
                        f"MŨI/MIỆNG | đã {elapsed:.1f}s, "
                        f"{countdown_txt}")
            else:
                votes_txt = ""
                if check_result is not None:
                    votes_txt = (f" | mũi {check_result.nose_votes_for_occluded}/4, "
                                 f"miệng {check_result.mouth_votes_for_occluded}/4")
                line = (f"{prefix} [{ts}] 🔴 ĐANG BỊ CHE MŨI/MIỆNG — "
                        f"đã {elapsed:.1f}s{votes_txt}, {countdown_txt}")
        else:
            line = f"{prefix} [{ts}] (state lạ: {state})"

        print(line)
        self._last_logged_state = state
        self._last_heartbeat_t  = now

    def _request_recalibrate(self, reason: str):
        print(f"🔄 Yêu cầu recalibrate: {reason}")
        self._recal_request.set()

    def _do_recalibrate(self):
        """Reset toàn bộ state để vào lại CALIBRATING."""
        self.detector.reset()
        self.smoother.reset()
        self.scene.reset()              # xây lại baseline độ nét
        self.calib_cond_warner.reset()  # quên cảnh báo điều kiện hiệu chỉnh
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
            last_alert_at = self.fsm.last_alert_at
            if last_alert_at is None:
                rem = max(0, OCCLUSION_THRESHOLD_SEC - elapsed)
                tail = f"Con lai: {rem:.1f}s"
            else:
                next_rem = max(
                    0.0,
                    OCCLUSION_THRESHOLD_SEC - (time.monotonic() - last_alert_at),
                )
                tail = f"Da gui | Gui lai sau: {next_rem:.1f}s"
            if trigger == TRIGGER_FACE_LOST:
                msg = f"MAT MAT - NGHI BI PHU KIN {elapsed:.1f}s | {tail}"
            else:
                msg = f"MUI/MIENG BI CHE {elapsed:.1f}s | {tail}"
            color = (0, 80, 255)
            cv2.putText(frame, ">>> CANH BAO CHE MUI/MIENG <<<",
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
        # Hâm nóng Telegram ngay từ đầu (thread nền) — lúc autostart máy vừa
        # boot mạng chưa lên; làm việc này song song để alert đầu gửi nhanh.
        self._warmup_telegram()

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
        # Autofocus ON mặc định — tắt nó là thủ phạm chính gây false alert khi
        # đối tượng đổi khoảng cách (ảnh mờ → edge/lap_var ~0 → tưởng bị che).
        cap.set(cv2.CAP_PROP_AUTOFOCUS,    1 if CAMERA_AUTOFOCUS else 0)

        actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print("🟢 Baby Monitor V5")
        print(f"   Detection mode   : {DETECTION_MODE.upper()}")
        if DETECTION_MODE == "strict":
            print(f"   → Strict mode: KHÔNG dùng multi-signal voting.")
            print(f"   → MediaPipe confidence cao ({MP_DETECTION_CONFIDENCE}) — face_lost trigger handle alert")
        print(f"   Camera           : {src} → {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
        print(f"   Autofocus        : {'BẬT' if CAMERA_AUTOFOCUS else 'TẮT'} "
              f"(CAMERA_AUTOFOCUS để đổi)")
        print(f"   Calibration      : {CALIBRATION_SEC}s")
        print(f"   Cảnh báo sau     : {OCCLUSION_THRESHOLD_SEC}s")
        print(f"   Auto-recal       : sau {AUTO_RECAL_AFTER_SEC}s safe liên tục"
              if AUTO_RECAL_AFTER_SEC > 0 else "   Auto-recal       : tắt")
        print(f"   Events lưu tại   : {EVENTS_DIR}")
        print(f"   Headless         : {HEADLESS}")
        print(f"   Log heartbeat    : mỗi {LOG_INTERVAL_SEC:.1f}s "
              f"(set LOG_INTERVAL_SEC để đổi)")
        print(f"   Recal manual     : nhấn R (GUI) hoặc `kill -USR1 <pid>` (headless)")
        print(f"   Blur-gate        : BẬT (drop_frac={BLUR_DROP_FRAC}) — chống false alert do mờ")
        print(f"   Watchdog         : BẬT (FPS<{HEALTH_MIN_FPS}/tối<{HEALTH_DARK_LUMA}/đơ → cảnh báo sau {HEALTH_DEGRADE_SEC:.0f}s)")
        print(f"   Heartbeat TG     : "
              + (f"mỗi {HEARTBEAT_SEC}s" if HEARTBEAT_SEC > 0 else "TẮT (đặt HEARTBEAT_SEC để bật)"))
        print(f"   Startup notify   : "
              + (f"BẬT (yêu cầu đưa mặt vào khung; nhắc lại mỗi {CALIB_REMIND_SEC}s)"
                 if STARTUP_NOTIFY else "TẮT"))
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

                # Thông báo khởi động về Telegram (yêu cầu đưa mặt vào khung).
                if STARTUP_NOTIFY:
                    if calib_done:   # strict mode — không cần hiệu chỉnh
                        self._calib_done_notified = True
                        self._notify_text("🟢 *Baby Monitor đã khởi động* (chế độ strict) — đang giám sát.")
                    else:
                        self._notify_text(
                            "🟢 *Baby Monitor đã khởi động.*\n"
                            "👉 Vui lòng đưa mặt trẻ vào khung camera để hệ thống "
                            "hiệu chỉnh (~5 giây). Em sẽ báo lại khi bắt đầu giám sát."
                        )

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

                    # === Phân tích frame mức toàn cục (blur-gate + watchdog) ===
                    # Rẻ CPU (~5ms), chạy trước MediaPipe để dùng cho cả frame này.
                    self.scene.analyze(frame)
                    fps = self._update_fps(now)
                    self._update_health(now, fps)

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
                        # === Cổng chất lượng calibration ===
                        # Baseline là "định nghĩa mặt sạch" mà mọi lần phát hiện
                        # che về sau so sánh với. Học baseline từ frame tối/mờ →
                        # baseline rác → kém tin cậy ngoài thực địa. Vì vậy CHỈ
                        # gom mẫu từ frame ĐẠT (đủ sáng + không mờ).
                        too_dark   = self.scene.luma < HEALTH_DARK_LUMA
                        too_blurry = self.scene.is_blurry
                        if not too_dark and not too_blurry:
                            self.detector.add_calibration_sample(frame, lms, w, h)
                            self.scene.update_sharp_baseline()
                        # Headless: điều kiện kém kéo dài → Telegram hướng dẫn fix.
                        self._maybe_warn_calib_conditions(now, too_dark, too_blurry)
                        if now - calib_start >= CALIBRATION_SEC:
                            ok, msg = self.detector.finalize_calibration()
                            if ok:
                                calib_done = True
                                self.calib_cond_warner.reset()
                                print(f"✅ Calibration xong! {msg}\n"
                                      f"   scene: sharp_baseline={self.scene.sharp_baseline:.0f}")
                                self._notify_calibrated()   # xác nhận "bắt đầu giám sát" về Telegram
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
                                    texture_reliable=not self.scene.is_blurry,
                                )
                                if check_result is not None:
                                    occluded_by_detector = self.smoother.update(
                                        check_result.occluded
                                    )
                                    # === Quick-confirm tín hiệu cực mạnh ===
                                    # Kịch bản che mũi+miệng đồng thời (tay/
                                    # khẩu trang/giấy nhỏ — mắt vẫn hở):
                                    # MediaPipe vẫn thấy mặt nhưng confidence
                                    # dao động → smoother 10-frame có thể
                                    # không kịp confirm trước khi mặt flicker
                                    # mất. Bypass khi cả 2 patch đều vote
                                    # rất rõ (≥3/4): 8/8 tín hiệu nói có che
                                    # → không cần 10-frame consensus, set
                                    # state ngay.
                                    if (check_result.nose_votes_for_occluded >= 3
                                            and check_result.mouth_votes_for_occluded >= 3):
                                        occluded_by_detector = True
                                        self.smoother.state = True
                        else:
                            # KHÔNG reset smoother khi face mất ngắn.
                            # Lý do: khi bị che mũi+miệng, MediaPipe có thể
                            # flicker thấy/mất → reset hard sẽ never confirm
                            # vì mỗi lần face quay lại phải đếm 10 frame từ
                            # đầu. Giữ state cũ → vote tích lũy được giữ lại.
                            # Khi face mất hoàn toàn → FSM face_lost path
                            # sẽ tự alert sau 15s, không phụ thuộc smoother.
                            pass

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
                            # Alert đầu của event (occlusion_start mới) → lưu ảnh.
                            # Re-alert cùng event → chỉ gửi Telegram + log.
                            is_first_alert = (
                                self.fsm.occlusion_start
                                != self._last_saved_occlusion_start
                            )
                            if is_first_alert:
                                self._last_saved_occlusion_start = (
                                    self.fsm.occlusion_start
                                )
                            kind = "CẢNH BÁO ĐẦU" if is_first_alert else "RE-ALERT"
                            print(f"🚨 Đủ ngưỡng {OCCLUSION_THRESHOLD_SEC}s "
                                  f"({kind}) — trigger={trigger} "
                                  f"elapsed={elapsed:.1f}s"
                                  + ("" if is_first_alert else " (không lưu ảnh trùng)"))
                            self._dispatch_alert(frame, elapsed,
                                                 check_result, trigger,
                                                 save_image=is_first_alert)

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

                    # Cảnh CLEAR → cập nhật baseline độ nét cho blur-gate.
                    if state == STATE_SAFE:
                        self.scene.update_sharp_baseline()

                    # === Heartbeat "vẫn đang canh" ===
                    self._maybe_heartbeat(now, state)
                    # Nhắc đưa mặt vào khung nếu mãi chưa hiệu chỉnh được.
                    self._maybe_remind_calibration(now, calib_done, face_present)

                    # === Log tiếng Việt cho headless (Pi 4) ===
                    self._log_status(
                        now=now, state=state, elapsed=elapsed,
                        trigger=trigger, check_result=check_result,
                        person_seen=person_seen,
                        calib_remaining=calib_remaining,
                        face_present=face_present,
                    )

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
