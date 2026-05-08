import cv2
import mediapipe as mp
import numpy as np
import time, asyncio, threading, io
from datetime import datetime
from telegram import Bot

# ===================== CONFIG =====================
TELEGRAM_TOKEN          = "8684958351:AAHE0XavsY_DgzEevmtcMUHB3_N3QwuNYIk"
CHAT_ID                 = "7316578932"
OCCLUSION_THRESHOLD_SEC = 15
COOLDOWN_SEC            = 60

CALIBRATION_SEC         = 5      # Giây calibration lúc đầu
HIST_CORR_THRESHOLD     = 0.65   # Correlation < này = bị che (0.0~1.0)
CONFIRM_FRAMES          = 15     # Frame liên tiếp để xác nhận bị che
BASELINE_UPDATE_RATE    = 0.01   # Cập nhật baseline chậm để thích nghi ánh sáng
PATCH_SIZE              = 50     # Kích thước patch lấy mẫu (px)
# ==================================================

mp_face_mesh = mp.solutions.face_mesh

# Nose tip landmark
NOSE_TIP = 4
# Mouth center landmark  
MOUTH_CENTER = 14
# Upper lip
UPPER_LIP = 0


def extract_patch(frame, landmark, w, h, size=PATCH_SIZE):
    """Cắt patch vuông xung quanh landmark, chuẩn hóa kích thước"""
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
    """Tính histogram HSV, normalize"""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [36, 32],        # H: 36 bins, S: 32 bins
        [0, 180, 0, 256]
    )
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def histogram_correlation(h1, h2):
    """So sánh 2 histogram: 1.0 = giống nhau, 0.0 = khác nhau"""
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


class BaselineManager:
    """
    Quản lý baseline: lưu histogram mũi/miệng khi an toàn.
    Cập nhật chậm để thích nghi với thay đổi ánh sáng tự nhiên.
    """
    def __init__(self):
        self.nose_hist     = None
        self.mouth_hist    = None
        self.is_ready      = False
        self._samples_nose  = []
        self._samples_mouth = []

    def add_calibration_sample(self, frame, landmarks, w, h):
        """Thu thập mẫu trong giai đoạn calibration"""
        nose_patch  = extract_patch(frame, landmarks[NOSE_TIP],    w, h)
        mouth_patch = extract_patch(frame, landmarks[MOUTH_CENTER], w, h)

        if nose_patch is not None:
            self._samples_nose.append(calc_histogram(nose_patch))
        if mouth_patch is not None:
            self._samples_mouth.append(calc_histogram(mouth_patch))

    def finalize_calibration(self):
        """Tính baseline từ các mẫu đã thu thập"""
        if not self._samples_nose or not self._samples_mouth:
            return False

        # Average histogram từ tất cả mẫu
        self.nose_hist  = np.mean(self._samples_nose,  axis=0).astype(np.float32)
        self.mouth_hist = np.mean(self._samples_mouth, axis=0).astype(np.float32)
        self.is_ready   = True
        return True

    def update(self, frame, landmarks, w, h):
        """
        Cập nhật baseline rất chậm khi đang an toàn.
        Giúp thích nghi với thay đổi ánh sáng từ từ (mây, đèn...).
        """
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
        """
        So sánh histogram hiện tại với baseline.
        Trả về (is_occluded, nose_corr, mouth_corr)
        """
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

        # Bị che khi CẢ MŨI VÀ MIỆNG đều có correlation thấp
        nose_occ  = nose_corr  < HIST_CORR_THRESHOLD
        mouth_occ = mouth_corr < HIST_CORR_THRESHOLD
        return nose_occ and mouth_occ, nose_corr, mouth_corr


class SmoothingBuffer:
    def __init__(self, confirm=CONFIRM_FRAMES, clear=8):
        self.buf   = []
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


class BabyMonitorV5:
    def __init__(self):
        self.bot             = Bot(token=TELEGRAM_TOKEN)
        self.baseline        = BaselineManager()
        self.smoother        = SmoothingBuffer()
        self.occlusion_start = None
        self.alert_sent      = False
        self.last_alert_time = 0

    def draw_ui(self, frame, landmarks, state, elapsed,
                face_present, nose_corr, mouth_corr,
                calib_remaining, w, h):

        # Vẽ landmarks mũi/miệng
        if landmarks and face_present and self.baseline.is_ready:
            nose_lm  = landmarks[NOSE_TIP]
            mouth_lm = landmarks[MOUTH_CENTER]

            nose_c  = (0,0,255) if nose_corr  < HIST_CORR_THRESHOLD else (0,255,0)
            mouth_c = (0,0,255) if mouth_corr < HIST_CORR_THRESHOLD else (0,255,0)

            half = PATCH_SIZE // 2
            nx = int(nose_lm.x * w);  ny = int(nose_lm.y * h)
            mx = int(mouth_lm.x * w); my = int(mouth_lm.y * h)

            cv2.rectangle(frame,
                          (nx-half, ny-half), (nx+half, ny+half), nose_c, 2)
            cv2.putText(frame, f"Mui {nose_corr:.2f}",
                        (nx-half, ny-half-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, nose_c, 1)

            cv2.rectangle(frame,
                          (mx-half, my-half), (mx+half, my+half), mouth_c, 2)
            cv2.putText(frame, f"Mieng {mouth_corr:.2f}",
                        (mx-half, my-half-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, mouth_c, 1)

        # Debug info
        debug = [
            f"Mui   corr: {nose_corr:.3f}  {'BI CHE' if nose_corr  < HIST_CORR_THRESHOLD else 'OK'}",
            f"Mieng corr: {mouth_corr:.3f}  {'BI CHE' if mouth_corr < HIST_CORR_THRESHOLD else 'OK'}",
            f"Nguong: {HIST_CORR_THRESHOLD}  (giam neu bao nham, tang neu bo sot)",
        ]
        for i, line in enumerate(debug):
            cv2.putText(frame, line, (10, 50 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1)

        # Border nguy hiểm
        if state == "ALERT":
            cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,220), 6)

        # Status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-65), (w, h), (15,15,15), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        if state == "CALIBRATING":
            msg   = f"Calibration... Giu nguyen mat tre {calib_remaining:.1f}s"
            color = (0, 200, 255)
            # Progress bar
            progress = 1.0 - (calib_remaining / CALIBRATION_SEC)
            cv2.rectangle(frame, (0, h-5), (int(w*progress), h), (0,200,255), -1)
        elif state == "NO_FACE":
            msg   = "Khong phat hien khuon mat"
            color = (100, 100, 100)
        elif state == "ALERT":
            rem = max(0, OCCLUSION_THRESHOLD_SEC - elapsed)
            msg   = f"MUI/MIENG BI CHE {elapsed:.1f}s | Con lai: {rem:.1f}s"
            color = (0, 80, 255)
            cv2.putText(frame, ">>> NGUY CO NGAT THO <<<",
                        (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,60,255), 2)
        else:  # SAFE
            msg   = "Binh thuong - Mui/mieng dang nhin thay"
            color = (0, 220, 0)

        cv2.putText(frame, msg, (10, h-35 if state == "ALERT" else h-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Timestamp + indicator
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        dot_color = {
            "CALIBRATING": (0,200,255),
            "SAFE":        (0,255,0),
            "ALERT":       (0,0,255),
            "NO_FACE":     (128,128,128),
        }.get(state, (128,128,128))
        cv2.circle(frame, (w-20, 20), 10, dot_color, -1)

        return frame

    async def send_alert(self, frame, elapsed, nose_corr, mouth_corr):
        now = time.time()
        if now - self.last_alert_time < COOLDOWN_SEC:
            return
        self.last_alert_time = now

        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        _, buf    = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img_io    = io.BytesIO(buf.tobytes())
        img_io.name = 'alert.jpg'

        caption = (
            f"🚨 *CẢNH BÁO NGẠT THỞ!*\n\n"
            f"⏰ Thời gian: `{ts}`\n"
            f"⏱ Bị che: `{elapsed:.1f} giây`\n\n"
            f"👃 Mũi correlation: `{nose_corr:.3f}`\n"
            f"👄 Miệng correlation: `{mouth_corr:.3f}`\n"
            f"📏 Ngưỡng: `{HIST_CORR_THRESHOLD}`\n\n"
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

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("🟢 Baby Monitor V5 - Histogram Baseline Correlation")
        print(f"   Calibration      : {CALIBRATION_SEC}s")
        print(f"   Corr threshold   : {HIST_CORR_THRESHOLD}")
        print(f"   Cảnh báo sau     : {OCCLUSION_THRESHOLD_SEC}s")
        print("\n⏳ Đang chờ mặt trẻ để calibrate...\n")

        calib_start    = None
        calib_done     = False
        nose_corr      = 1.0
        mouth_corr     = 1.0

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

                state          = "NO_FACE"
                calib_remaining = 0.0

                if face_present:
                    lms = res.multi_face_landmarks[0].landmark

                    # --- CALIBRATION PHASE ---
                    if not calib_done:
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
                                calib_start = None  # Thử lại

                        state = "CALIBRATING"

                    # --- MONITORING PHASE ---
                    else:
                        raw_occ, nose_corr, mouth_corr = \
                            self.baseline.check_occlusion(frame, lms, w, h)
                        occluded = self.smoother.update(raw_occ)

                        if occluded:
                            state = "ALERT"
                            if self.occlusion_start is None:
                                self.occlusion_start = now
                                print(f"⚠️  Bị che! nose={nose_corr:.2f} mouth={mouth_corr:.2f}")
                            elapsed = now - self.occlusion_start

                            if elapsed >= OCCLUSION_THRESHOLD_SEC and not self.alert_sent:
                                self.alert_sent = True
                                _f = frame.copy()
                                _e, _n, _m = elapsed, nose_corr, mouth_corr
                                threading.Thread(
                                    target=lambda: asyncio.run(
                                        self.send_alert(_f, _e, _n, _m)
                                    ), daemon=True
                                ).start()
                        else:
                            if self.occlusion_start:
                                print("✅ Bình thường trở lại")
                            self.occlusion_start = None
                            self.alert_sent      = False
                            elapsed              = 0.0
                            state                = "SAFE"
                            # Cập nhật baseline khi đang an toàn
                            self.baseline.update(frame, lms, w, h)
                else:
                    # Không thấy mặt → reset
                    calib_start = None if not calib_done else calib_start
                    self.smoother.reset()
                    self.occlusion_start = None
                    self.alert_sent      = False
                    elapsed              = 0.0
                    lms                  = None

                lms_draw = (res.multi_face_landmarks[0].landmark
                            if face_present else None)
                frame = self.draw_ui(
                    frame, lms_draw, state, elapsed if state == "ALERT" else 0.0,
                    face_present, nose_corr, mouth_corr, calib_remaining, w, h
                )

                cv2.imshow('Baby Monitor V5 | Q = thoat', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        print("🔴 Đã dừng")


if __name__ == "__main__":
    BabyMonitorV5().run()