import cv2
import mediapipe as mp
import numpy as np
import time, asyncio, threading, io, json, os
from datetime import datetime
from pathlib import Path
from telegram import Bot

from state_machine import (
    OcclusionStateMachine,
    STATE_ALERT, STATE_SAFE, STATE_NO_FACE, STATE_CALIBRATING,
    TRIGGER_FACE_LOST,
)

# ===================== CONFIG =====================
TELEGRAM_TOKEN = os.environ.get(
    "TELEGRAM_TOKEN",
    "8684958351:AAHE0XavsY_DgzEevmtcMUHB3_N3QwuNYIk",
)
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7316578932")

OCCLUSION_THRESHOLD_SEC = 15
COOLDOWN_SEC            = 60

CALIBRATION_SEC         = 5
HIST_CORR_THRESHOLD     = 0.65
CONFIRM_FRAMES          = 15
BASELINE_UPDATE_RATE    = 0.01
PATCH_SIZE              = 50

FACE_LOST_GRACE_SEC     = 1.5
FACE_GONE_RESET_SEC     = OCCLUSION_THRESHOLD_SEC + 10

# YOLO settings (optional). Nếu không cài ultralytics → bỏ qua, vẫn chạy được.
YOLO_PERSON_CONF        = 0.35
# Auto-throttle: nếu có CUDA, chạy YOLO mỗi 2 frame (~15 FPS); CPU thì mỗi 5
# frame (~6 FPS). Có thể override qua env YOLO_EVERY.
YOLO_RUN_EVERY_N_FRAMES = int(os.environ.get("YOLO_EVERY", "0")) or None
YOLO_CACHE_TTL_SEC      = 1.0
YOLO_DEVICE             = os.environ.get("YOLO_DEVICE", "auto")  # auto|cuda|cpu

# Camera settings — MJPG cho FPS cao, buffer=1 cho latency thấp
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "0")
CAMERA_WIDTH  = int(os.environ.get("CAMERA_WIDTH",  "1280"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT",  "720"))
CAMERA_FPS    = int(os.environ.get("CAMERA_FPS",     "30"))

PROJECT_ROOT     = Path(__file__).resolve().parent.parent
EVENTS_DIR       = PROJECT_ROOT / "events"
EVENTS_DIR.mkdir(exist_ok=True)
YOLO_MODEL_PATH  = PROJECT_ROOT / "yolov8n.pt"
# ==================================================

mp_face_mesh = mp.solutions.face_mesh

NOSE_TIP     = 4
MOUTH_CENTER = 14


def extract_patch(frame, landmark, w, h, size=PATCH_SIZE):
    x = int(np.clip(landmark.x * w, 0, w - 1))
    y = int(np.clip(landmark.y * h, 0, h - 1))
    half = size // 2
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (size, size))


def calc_histogram(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [36, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def histogram_correlation(h1, h2):
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


class BaselineManager:
    def __init__(self):
        self.nose_hist      = None
        self.mouth_hist     = None
        self.is_ready       = False
        self._samples_nose  = []
        self._samples_mouth = []

    def add_calibration_sample(self, frame, landmarks, w, h):
        nose_patch  = extract_patch(frame, landmarks[NOSE_TIP],    w, h)
        mouth_patch = extract_patch(frame, landmarks[MOUTH_CENTER], w, h)
        if nose_patch is not None:
            self._samples_nose.append(calc_histogram(nose_patch))
        if mouth_patch is not None:
            self._samples_mouth.append(calc_histogram(mouth_patch))

    def finalize_calibration(self):
        if not self._samples_nose or not self._samples_mouth:
            return False
        self.nose_hist  = np.mean(self._samples_nose,  axis=0).astype(np.float32)
        self.mouth_hist = np.mean(self._samples_mouth, axis=0).astype(np.float32)
        self.is_ready   = True
        return True

    def update(self, frame, landmarks, w, h):
        if not self.is_ready:
            return
        nose_patch  = extract_patch(frame, landmarks[NOSE_TIP],    w, h)
        mouth_patch = extract_patch(frame, landmarks[MOUTH_CENTER], w, h)
        if nose_patch is not None:
            curr_hist = calc_histogram(nose_patch).astype(np.float32)
            self.nose_hist = ((1 - BASELINE_UPDATE_RATE) * self.nose_hist
                              + BASELINE_UPDATE_RATE * curr_hist)
        if mouth_patch is not None:
            curr_hist = calc_histogram(mouth_patch).astype(np.float32)
            self.mouth_hist = ((1 - BASELINE_UPDATE_RATE) * self.mouth_hist
                               + BASELINE_UPDATE_RATE * curr_hist)

    def check_occlusion(self, frame, landmarks, w, h):
        if not self.is_ready:
            return False, 1.0, 1.0
        nose_patch  = extract_patch(frame, landmarks[NOSE_TIP],    w, h)
        mouth_patch = extract_patch(frame, landmarks[MOUTH_CENTER], w, h)
        if nose_patch is None or mouth_patch is None:
            return False, 1.0, 1.0
        nose_corr  = histogram_correlation(
            calc_histogram(nose_patch).astype(np.float32),
            self.nose_hist
        )
        mouth_corr = histogram_correlation(
            calc_histogram(mouth_patch).astype(np.float32),
            self.mouth_hist
        )
        nose_occ  = nose_corr  < HIST_CORR_THRESHOLD
        mouth_occ = mouth_corr < HIST_CORR_THRESHOLD
        # OR-logic: chỉ một bên bị che cũng đáng lo
        return nose_occ or mouth_occ, nose_corr, mouth_corr


class SmoothingBuffer:
    def __init__(self, confirm=CONFIRM_FRAMES, clear=8):
        self.buf     = []
        self.confirm = confirm
        self.clear   = clear
        self.state   = False

    def update(self, val: bool) -> bool:
        self.buf.append(val)
        if len(self.buf) > max(self.confirm, self.clear):
            self.buf.pop(0)
        if len(self.buf) >= self.confirm and all(self.buf[-self.confirm:]):
            self.state = True
        if len(self.buf) >= self.clear and not any(self.buf[-self.clear:]):
            self.state = False
        return self.state

    def reset(self):
        self.buf   = []
        self.state = False


def _resolve_yolo_device(requested: str) -> str:
    """Auto-detect device. Trả về 'cuda:0' / 'cpu'."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    if requested == "cuda":
        print("⚠️  Yêu cầu cuda nhưng torch.cuda không available → fallback cpu")
    return "cpu"


class PersonDetector:
    """YOLO wrapper — phân biệt 'trẻ rời khung' và 'trẻ vẫn ở đó nhưng bị phủ kín'.

    Optional: nếu không cài ultralytics hoặc model không tồn tại → has_person()
    trả về None, state machine sẽ tự rơi về heuristic theo thời gian.

    Tự dùng CUDA nếu có. Có thể ép qua env YOLO_DEVICE=cpu/cuda.
    """

    def __init__(self, model_path: Path):
        self.model        = None
        self.device       = "cpu"
        self.run_every    = 5
        self._last_call_t = 0.0
        self._last_result = None  # type: bool | None
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
            # Throttle: nếu user override → dùng; không thì auto theo device
            if YOLO_RUN_EVERY_N_FRAMES is not None:
                self.run_every = YOLO_RUN_EVERY_N_FRAMES
            else:
                self.run_every = 2 if self.device.startswith("cuda") else 5
            print(f"✅ YOLO sẵn sàng: {model_path.name} | "
                  f"device={self.device} | run_every={self.run_every}")
        except Exception as e:
            print(f"⚠️  Lỗi load YOLO: {e}; YOLO bị tắt.")
            self.model = None

    def has_person(self, frame) -> bool | None:
        """Trả về True/False/None. Throttle theo FPS + cache TTL."""
        if self.model is None:
            return None
        now = time.time()
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
        self.baseline        = BaselineManager()
        self.smoother        = SmoothingBuffer()
        self.person_detector = PersonDetector(YOLO_MODEL_PATH)
        self.fsm             = OcclusionStateMachine(
            threshold_sec  = OCCLUSION_THRESHOLD_SEC,
            grace_sec      = FACE_LOST_GRACE_SEC,
            gone_reset_sec = FACE_GONE_RESET_SEC,
        )
        self.last_alert_time = 0.0

    # ---------- Event persistence ----------
    def _save_event(self, frame, status, nose_corr, mouth_corr, trigger):
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_iso  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            jpg_path  = EVENTS_DIR / f"possible_suffocation_risk_{ts_file}.jpg"
            json_path = EVENTS_DIR / f"possible_suffocation_risk_{ts_file}.json"
            cv2.imwrite(str(jpg_path), frame)
            payload = {
                "event_type": "possible_suffocation_risk",
                "time": ts_iso,
                "extra": {
                    "status": status,
                    "lower_score": float(mouth_corr),
                    "upper_score": float(nose_corr),
                    "alert_seconds": OCCLUSION_THRESHOLD_SEC,
                    "trigger": trigger,
                },
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"💾 Đã lưu event: {jpg_path.name}")
        except Exception as e:
            print(f"❌ Lỗi lưu event: {e}")

    # ---------- Telegram alert ----------
    async def send_alert(self, frame, elapsed, nose_corr, mouth_corr, trigger):
        now = time.time()
        if now - self.last_alert_time < COOLDOWN_SEC:
            return
        self.last_alert_time = now

        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_io = io.BytesIO(buf.tobytes())
        img_io.name = 'alert.jpg'

        if trigger == TRIGGER_FACE_LOST:
            reason_block = "🫥 Mất hoàn toàn khuôn mặt — nghi bị phủ kín bởi vật lạ"
        else:
            reason_block = (
                f"👃 Mũi correlation: `{nose_corr:.3f}`\n"
                f"👄 Miệng correlation: `{mouth_corr:.3f}`\n"
                f"📏 Ngưỡng: `{HIST_CORR_THRESHOLD}`"
            )

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

    def _dispatch_alert(self, frame, elapsed, nose_corr, mouth_corr, trigger):
        status = ("FULLY_COVERED" if trigger == TRIGGER_FACE_LOST
                  else "POSSIBLE_OCCLUSION")
        snap = frame.copy()
        self._save_event(snap, status, nose_corr, mouth_corr, trigger)
        threading.Thread(
            target=lambda: asyncio.run(
                self.send_alert(snap, elapsed, nose_corr, mouth_corr, trigger)
            ), daemon=True
        ).start()

    # ---------- UI ----------
    def draw_ui(self, frame, landmarks, state, elapsed,
                face_present, nose_corr, mouth_corr,
                calib_remaining, w, h, trigger="", person_seen=None):

        if landmarks and face_present and self.baseline.is_ready:
            nose_lm  = landmarks[NOSE_TIP]
            mouth_lm = landmarks[MOUTH_CENTER]
            nose_c  = (0,0,255) if nose_corr  < HIST_CORR_THRESHOLD else (0,255,0)
            mouth_c = (0,0,255) if mouth_corr < HIST_CORR_THRESHOLD else (0,255,0)
            half = PATCH_SIZE // 2
            nx = int(nose_lm.x * w);  ny = int(nose_lm.y * h)
            mx = int(mouth_lm.x * w); my = int(mouth_lm.y * h)
            cv2.rectangle(frame, (nx-half, ny-half), (nx+half, ny+half), nose_c, 2)
            cv2.putText(frame, f"Mui {nose_corr:.2f}", (nx-half, ny-half-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, nose_c, 1)
            cv2.rectangle(frame, (mx-half, my-half), (mx+half, my+half), mouth_c, 2)
            cv2.putText(frame, f"Mieng {mouth_corr:.2f}", (mx-half, my-half-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, mouth_c, 1)

        person_label = (
            "YOLO: nguoi" if person_seen is True
            else "YOLO: khong nguoi" if person_seen is False
            else "YOLO: off"
        )
        debug = [
            f"Mui   corr: {nose_corr:.3f}  {'BI CHE' if nose_corr  < HIST_CORR_THRESHOLD else 'OK'}",
            f"Mieng corr: {mouth_corr:.3f}  {'BI CHE' if mouth_corr < HIST_CORR_THRESHOLD else 'OK'}",
            f"Nguong: {HIST_CORR_THRESHOLD}  | {person_label}",
        ]
        for i, line in enumerate(debug):
            cv2.putText(frame, line, (10, 50 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1)

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
            msg   = "Binh thuong - Mui/mieng dang nhin thay"
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
        # Camera source: int (USB index) hoặc string (RTSP URL / GStreamer pipeline)
        src = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"❌ Không mở được camera: {src}")
            return

        # MJPG → cho phép FPS cao ở 720p (YUYV thường giới hạn 10fps@720p)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        # Buffer 1 → giảm latency, frame mới nhất luôn được đọc
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Tắt autofocus — webcam baby monitor cần focus cố định
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print("🟢 Baby Monitor V5 - Histogram + State Machine + YOLO")
        print(f"   Camera           : {src} → {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
        print(f"   Calibration      : {CALIBRATION_SEC}s")
        print(f"   Corr threshold   : {HIST_CORR_THRESHOLD}")
        print(f"   Cảnh báo sau     : {OCCLUSION_THRESHOLD_SEC}s")
        print(f"   Events lưu tại   : {EVENTS_DIR}")
        print("\n⏳ Đang chờ mặt trẻ để calibrate...\n")

        calib_start = None
        calib_done  = False
        nose_corr   = 1.0
        mouth_corr  = 1.0

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as face_mesh:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                h, w  = frame.shape[:2]
                now   = time.time()
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = face_mesh.process(rgb)
                face_present = bool(res.multi_face_landmarks)

                state           = STATE_NO_FACE
                calib_remaining = 0.0
                elapsed         = 0.0
                trigger         = ""
                person_seen     = None

                # YOLO chỉ chạy khi đã calib xong VÀ không thấy mặt — tiết kiệm CPU
                # và chỉ cần khi ta cần phân biệt 'rời khung' vs 'bị phủ'.
                if calib_done and not face_present:
                    person_seen = self.person_detector.has_person(frame)

                if face_present and not calib_done:
                    # --- Giai đoạn calibration ---
                    lms = res.multi_face_landmarks[0].landmark
                    if calib_start is None:
                        calib_start = now
                        print("✅ Phát hiện mặt! Đang calibrate baseline...")
                    calib_remaining = max(0, CALIBRATION_SEC - (now - calib_start))
                    self.baseline.add_calibration_sample(frame, lms, w, h)
                    if now - calib_start >= CALIBRATION_SEC:
                        if self.baseline.finalize_calibration():
                            calib_done = True
                            print("✅ Calibration xong! Bắt đầu giám sát.\n")
                        else:
                            calib_start = None
                    state = STATE_CALIBRATING

                elif not face_present and not calib_done:
                    # Chưa calibrate, vẫn chờ thấy mặt
                    calib_start = None
                    state = STATE_NO_FACE

                else:
                    # --- Giai đoạn giám sát ---
                    occluded_by_hist = False
                    if face_present:
                        lms = res.multi_face_landmarks[0].landmark
                        raw_occ, nose_corr, mouth_corr = \
                            self.baseline.check_occlusion(frame, lms, w, h)
                        occluded_by_hist = self.smoother.update(raw_occ)
                    else:
                        # Không thấy mặt → smoother không có input mới, reset cho sạch
                        self.smoother.reset()

                    result = self.fsm.step(
                        now=now,
                        face_present=face_present,
                        occluded_by_histogram=occluded_by_hist,
                        person_in_frame=person_seen,
                    )
                    state   = result.state
                    elapsed = result.elapsed
                    trigger = result.trigger

                    # In log khi chuyển trạng thái nguy hiểm
                    if result.should_alert:
                        print(f"🚨 Đủ ngưỡng {OCCLUSION_THRESHOLD_SEC}s — "
                              f"trigger={trigger} elapsed={elapsed:.1f}s")
                        self._dispatch_alert(frame, elapsed,
                                             nose_corr, mouth_corr, trigger)

                    # Cập nhật baseline khi an toàn (chỉ khi có mặt)
                    if state == STATE_SAFE and face_present:
                        lms = res.multi_face_landmarks[0].landmark
                        self.baseline.update(frame, lms, w, h)

                lms_draw = (res.multi_face_landmarks[0].landmark
                            if face_present else None)
                frame = self.draw_ui(
                    frame, lms_draw, state, elapsed,
                    face_present, nose_corr, mouth_corr,
                    calib_remaining, w, h,
                    trigger=trigger, person_seen=person_seen,
                )

                cv2.imshow('Baby Monitor V5 | Q = thoat', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("🔴 Đã dừng")


if __name__ == "__main__":
    BabyMonitorV5().run()
