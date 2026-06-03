# Hướng dẫn cài đặt Baby AI Alert trên Raspberry Pi 4

Hướng dẫn **độc lập, chi tiết** cho board **Raspberry Pi 4 Model B** (Broadcom BCM2711, 4 nhân Cortex-A72, ARM64/aarch64, CPU-only). Đây là nền tảng **chính thức và duy nhất** của dự án — mọi default trong code đã được tune sẵn cho Pi 4.

> 📌 **Toàn bộ default đã tune sẵn cho Pi 4 trong [src/main.py](src/main.py):** 640×480 @15fps, `YOLO_EVERY=10`, `HEADLESS=1`. Chạy thẳng là đúng cấu hình — **không cần sửa code, không bắt buộc file `.env`** cho phần hiệu năng. File `.env` giờ chỉ cần để đặt **token Telegram thật** của bạn (xem [§7](#7-tạo-file-env--token-telegram)).

---

## Mục lục

1. [Phần cứng cần có](#1-phần-cứng-cần-có)
2. [Cài hệ điều hành (Raspberry Pi OS 64-bit)](#2-cài-hệ-điều-hành-raspberry-pi-os-64-bit)
3. [Setup ban đầu (SSH + update + giờ + swap)](#3-setup-ban-đầu)
4. [Cài system dependencies](#4-cài-system-dependencies)
5. [Cài Python + project dependencies](#5-cài-python--project-dependencies)
6. [Tạo Telegram bot + lấy token](#6-tạo-telegram-bot--lấy-token)
7. [Tạo file `.env` — token Telegram](#7-tạo-file-env--token-telegram)
8. [Verify cài đặt](#8-verify-cài-đặt)
9. [Chạy chương trình](#9-chạy-chương-trình)
10. [Autostart bằng systemd](#10-autostart-bằng-systemd)
11. [Tối ưu hiệu năng cho Pi 4](#11-tối-ưu-hiệu-năng-cho-pi-4)
12. [(Optional) Kích relay khi có cảnh báo](#12-optional-kích-relay-khi-có-cảnh-báo)
13. [(Optional) Tăng tốc YOLO bằng Google Coral USB TPU](#13-optional-tăng-tốc-yolo-bằng-google-coral-usb-tpu)
14. [Troubleshooting](#14-troubleshooting)
15. [Tổng kết stack](#15-tổng-kết-stack)

---

## 1. Phần cứng cần có

| Thành phần | Khuyến nghị cho Pi 4 |
|---|---|
| **Board** | Raspberry Pi 4 Model B — **4GB hoặc 8GB RAM** (1GB/2GB sẽ OOM lúc cài MediaPipe, xem [§3.4](#34-tạo-swap-bắt-buộc-nếu-ram--4gb)) |
| **Storage** | microSD ≥32GB **Class 10 / A1-A2** (nên dùng loại A2 cho I/O ổn). Production 24/7 → dùng **SSD USB 3.0** boot thay thẻ SD (xem [§2.5](#25-khuyến-nghị-production-boot-từ-ssd-usb)) |
| **Camera** | Logitech USB webcam (C270/C310/C920…). USB webcam đơn giản & ổn nhất. CSI camera (Pi Camera) cũng được nhưng cần thêm bước, xem [§4.4](#44-nếu-dùng-pi-camera-csi-thay-usb-webcam) |
| **Mạng** | Ethernet (ổn định nhất cho alert) hoặc WiFi |
| **Tản nhiệt** | **Bắt buộc heatsink + quạt.** Pi 4 chạy YOLO + MediaPipe 24/7 sẽ nóng → thermal throttle xuống ~600MHz nếu không tản nhiệt |
| **Nguồn** | **Adapter USB-C 5V/3A chính hãng (≥15W).** ⚠️ Pi 4 rất kén nguồn — cáp điện thoại 5V/2A sẽ gây undervoltage (biểu tượng tia chớp ⚡, throttle, USB chập chờn) |

### Verify model sau khi cài OS
```bash
cat /proc/device-tree/model
# Mong đợi: "Raspberry Pi 4 Model B Rev 1.x"

uname -m
# PHẢI ra: aarch64   (nếu ra armv7l → đang chạy OS 32-bit, phải flash lại — xem §2)
```

---

## 2. Cài hệ điều hành (Raspberry Pi OS 64-bit)

> ⚠️ **BẮT BUỘC 64-bit (aarch64).** MediaPipe **không phát hành wheel cho ARM 32-bit (armv7l)** → `pip install mediapipe` sẽ fail thẳng. Đừng dùng bản "Raspberry Pi OS (32-bit)".

Khuyến nghị **Raspberry Pi OS (64-bit), bản Bookworm** — có sẵn Python 3.11 (đúng dải MediaPipe cần là 3.9–3.11).

### 2.1 Flash bằng Raspberry Pi Imager

1. Tải **Raspberry Pi Imager**: https://www.raspberrypi.com/software/
2. Mở Imager → **CHOOSE DEVICE**: Raspberry Pi 4
3. **CHOOSE OS** → `Raspberry Pi OS (other)` → **Raspberry Pi OS (64-bit)**
   - Headless (chạy server không màn hình) → chọn bản **Lite (64-bit)** cho nhẹ RAM.
   - Có cắm HDMI để dev → bản **Full / Desktop (64-bit)** cũng được.
4. **CHOOSE STORAGE** → chọn thẻ SD / SSD.

### 2.2 Cấu hình sẵn trước khi ghi (rất nên làm — bật SSH + WiFi luôn)

Bấm bánh răng ⚙️ (hoặc **EDIT SETTINGS**) trong Imager:
- **Set hostname**: `babypi` (tuỳ ý)
- **Enable SSH** → *Use password authentication* (hoặc dán public key)
- **Set username and password**: vd user `pi`, mật khẩu riêng (Bookworm **không còn** user mặc định `pi/raspberry` — bạn tự đặt)
- **Configure wireless LAN**: SSID + mật khẩu WiFi + country `VN`
- **Set locale**: timezone `Asia/Ho_Chi_Minh`

→ Bấm **SAVE** → **WRITE**. Cách này giúp boot lên là SSH vào được ngay, không cần màn hình.

### 2.3 Boot lần đầu

1. Cắm thẻ SD/SSD vào Pi 4, cắm nguồn USB-C (và Ethernet nếu có).
2. Đợi 1–2 phút boot xong.
3. SSH vào từ máy khác:
   ```bash
   ssh pi@babypi.local        # hoặc ssh pi@<ip-cua-pi>
   ```
   (nếu `.local` không resolve, tra IP trong router hoặc cắm HDMI xem)

### 2.4 (Nếu KHÔNG cấu hình sẵn ở §2.2) Bật SSH thủ công
Cắm HDMI + bàn phím, đăng nhập, rồi:
```bash
sudo raspi-config
# Interface Options → SSH → Enable
```

### 2.5 (Khuyến nghị production) Boot từ SSD USB

Thẻ SD hay hỏng sau vài tháng ghi log/snapshot 24/7. Pi 4 boot thẳng từ SSD USB 3.0 được:
1. Flash OS vào SSD (gắn qua adapter USB-SATA / NVMe-USB) y như §2.1.
2. Cắm SSD vào cổng **USB 3.0 (màu xanh)**, rút thẻ SD ra, cắm nguồn.
3. Pi 4 (bootloader đời mới) tự boot từ USB. Nếu không, cập nhật bootloader:
   ```bash
   sudo raspi-config   # Advanced Options → Boot Order → USB Boot
   sudo rpi-eeprom-update -a && sudo reboot
   ```

---

## 3. Setup ban đầu

### 3.1 Kiểm tra mạng
```bash
ip a                 # xem IP
ping -c3 8.8.8.8     # có internet chưa (cần cho alert Telegram)
```

### 3.2 Update hệ thống
```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

### 3.3 Đồng bộ giờ (timestamp alert phải đúng)
```bash
sudo timedatectl set-timezone Asia/Ho_Chi_Minh
sudo timedatectl set-ntp true
timedatectl status        # verify "System clock synchronized: yes"
```

### 3.4 Tạo swap (BẮT BUỘC nếu RAM ≤ 4GB)

Build/cài `mediapipe` + `torch` ngốn RAM, dễ OOM (`Killed`) trên bản 2/4GB. Pi OS đã có sẵn dphys-swapfile — nâng nó lên 2GB:

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
free -h        # verify cột Swap ~2.0Gi
```

> Sau khi cài xong có thể hạ swap về 512MB lại nếu muốn đỡ ghi thẻ SD — nhưng để 2GB chạy 24/7 cũng không sao.

---

## 4. Cài system dependencies

### 4.1 Build tools + thư viện hệ thống
```bash
sudo apt install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential cmake pkg-config git \
    libopencv-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libv4l-dev v4l-utils \
    libatlas-base-dev gfortran \
    libffi-dev libssl-dev
```

> `libatlas-base-dev` quan trọng trên Pi: NumPy/OpenCV cần BLAS để chạy nhanh; thiếu nó MediaPipe có thể chậm bất thường.

### 4.2 Cho user quyền truy cập camera
```bash
sudo usermod -aG video $USER
# BẮT BUỘC logout/login lại (hoặc reboot) để áp dụng
```

### 4.3 Kiểm tra USB webcam được nhận

Cắm webcam vào cổng **USB 3.0 (màu xanh)** — không cắm USB 2.0 (đen) vì giới hạn băng thông → tụt FPS:
```bash
v4l2-ctl --list-devices
# Mong đợi thấy: "HD Webcam C270 (usb-...)" → /dev/video0, /dev/video1

v4l2-ctl -d /dev/video0 --list-formats-ext
# Liệt kê resolution + FPS camera hỗ trợ (đảm bảo có 640x480)
```

### 4.4 (Nếu dùng Pi Camera CSI thay USB webcam)

Bookworm dùng `libcamera`. Để OpenCV (V4L2) đọc được CSI cam, bật legacy/V4L2 bridge:
```bash
# Kiểm tra camera CSI nhận chưa
libcamera-hello --list-cameras

# Thường libcamera tự tạo /dev/video0 cho CSI. Nếu main.py không mở được,
# đặt CAMERA_SOURCE trong .env trỏ đúng index (thử 0, rồi /dev/video0).
```
> Đơn giản nhất cho dự án này vẫn là **USB webcam** (đã test, plug-and-play). CSI cam cần fiddle thêm — chỉ dùng nếu bạn đã quen libcamera.

---

## 5. Cài Python + project dependencies

### 5.1 Lấy source code
```bash
# Đặt vào home cho gọn (đâu cũng được — app dùng path tương đối từ __file__)
cd ~
git clone <your-repo-url> baby-ai-alert
cd baby-ai-alert
```

> Các bước sau giả định bạn đang ở trong thư mục project (`~/baby-ai-alert`). Gọi nó là `$PROJECT_DIR`.

### 5.2 Verify Python 3.11 (Bookworm có sẵn)
```bash
python3 --version
# Mong đợi: Python 3.11.x
```
Nếu là **Python 3.12+** (bản OS rất mới): MediaPipe có thể chưa có wheel → cài 3.11 song song:
```bash
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```
và dùng `python3.11 -m venv venv` ở bước sau.

### 5.3 Cài 1 phát ăn ngay (KHUYẾN NGHỊ)

Dự án có sẵn script tự xử lý mọi cạm bẫy ARM64 CPU-only (torch kéo nvidia-cudnn vô dụng, pip cache hỏng, numpy/opencv mismatch…):

```bash
bash scripts/install_pi4.sh
```

Script tự động:
1. Set `TMPDIR=$HOME/tmp` (trên Pi OS `/tmp` nằm trên thẻ SD nên thường không đầy, nhưng set cho chắc — vô hại).
2. Tạo `venv/` nếu chưa có + activate.
3. Upgrade pip + wheel + setuptools.
4. Uninstall mọi `torch`/`nvidia-*`/`cuda-toolkit`/`triton` cũ (nếu lỡ cài).
5. `pip cache purge`.
6. Cài `torch>=2.0,<2.4` **TRƯỚC** (chặn ultralytics kéo torch ≥2.4 → nvidia-cudnn 433MB).
7. Cài phần còn lại từ `requirements.txt`.
8. Verify numpy<2 / opencv<4.11 + smoke test import tất cả lib.

Báo `✅ CÀI XONG` → sang [§6](#6-tạo-telegram-bot--lấy-token).

### 5.4 (Alternative) Cài thủ công từng bước

Chỉ làm khi muốn hiểu rõ hoặc đang debug.

**5.4.1 Tạo venv** (BẮT BUỘC — Bookworm chặn `pip install` vào Python hệ thống theo PEP 668 / "externally-managed-environment"):
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

**5.4.2 Cài torch CPU-only TRƯỚC** (quan trọng):
```bash
pip install --no-cache-dir 'torch>=2.0,<2.4'
```
> 🚫 Nếu thấy pip download `nvidia_cudnn_cu13-*.whl` (433MB) / `cuda_toolkit-*.whl` → **Ctrl+C ngay**. Pin torch không có hiệu lực → `git pull` lấy `requirements.txt` mới.

**5.4.3 Cài phần còn lại:**
```bash
pip install --no-cache-dir -r requirements.txt
```

`requirements.txt` đã pin sẵn để tránh các bẫy ABI trên ARM64:
- `numpy>=1.24,<2` — mediapipe 0.10.x compile chống NumPy 1.x
- `opencv-python>=4.8,<4.11` — opencv ≥4.11 compile chống NumPy 2.x
- `torch>=2.0,<2.4` — torch ≥2.4 kéo `nvidia-cudnn-cu13` ~433MB dù CPU-only
- `ultralytics>=8.0,<8.3` — ultralytics ≥8.3 yêu cầu torch ≥2.4
- `mediapipe>=0.10`, `python-telegram-bot>=20`

### 5.5 Verify + fix version conflicts
```bash
bash scripts/fix_env.sh
python -c "import cv2, mediapipe, numpy, telegram, ultralytics, torch; print('All imports OK')"
```
Phải in `All imports OK`. Nếu fail → [§14 Troubleshooting](#14-troubleshooting).

> `src/main.py` còn có 3 lớp guard tự refuse start nếu phát hiện numpy≥2 / opencv≥4.11 / nhiều opencv variant — không thể silent corruption.

---

## 6. Tạo Telegram bot + lấy token

### 6.1 Tạo bot
1. Mở Telegram, tìm `@BotFather` → gõ `/newbot`.
2. Đặt tên + username (kết thúc bằng `bot`).
3. Copy **token** dạng `1234567890:ABC...`.

### 6.2 Lấy chat ID
1. Gõ vài tin nhắn cho bot vừa tạo.
2. Mở: `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Tìm `"chat":{"id":7316578932,...}` → đó là **chat ID**.

### 6.3 Test gửi từ chính Pi 4
```bash
TOKEN="..."
CHAT="..."
curl -s "https://api.telegram.org/bot$TOKEN/sendMessage?chat_id=$CHAT&text=test+from+pi4"
```
Phải nhận được tin "test from pi4" trong Telegram.

---

## 7. Tạo file `.env` — token Telegram

> **`.env` còn cần làm gì:** Mọi default hiệu năng (640×480 @15fps, `YOLO_EVERY=10`, `HEADLESS=1`) đã hardcode sẵn cho Pi 4 trong [src/main.py](src/main.py) — **không cần `.env` cho phần này nữa**. Việc thật sự cần là đặt **token Telegram thật** của bạn (token trong code chỉ là demo). Vì vậy vẫn nên tạo `.env`.

Dự án có sẵn file mẫu — copy và sửa token:
```bash
cp .env.example .env
nano .env            # đổi TELEGRAM_TOKEN + TELEGRAM_CHAT_ID thành của bạn
chmod 600 .env       # bảo vệ token (chỉ owner đọc được)
```

`.env.example` tối thiểu chỉ cần 2 dòng:
```ini
TELEGRAM_TOKEN=<token-bot-cua-ban>
TELEGRAM_CHAT_ID=<chat-id-cua-ban>
```

### Bảng đầy đủ các biến (default trong code = đã tune cho Pi 4)

| Biến | Default code (Pi 4) | Tác dụng |
|---|---|---|
| `TELEGRAM_TOKEN` | token demo | **Đổi thành token của bạn** |
| `TELEGRAM_CHAT_ID` | chat demo | **Đổi thành chat ID của bạn** |
| `CAMERA_SOURCE` | `0` | index/`/dev/video0`/`rtsp://...` |
| `CAMERA_WIDTH` / `HEIGHT` / `FPS` | **640 / 480 / 15** | Camera đặt xa → thử 800×600. Đừng lên 720p (Pi 4 tụt FPS) |
| `YOLO_EVERY` | **`10`** | YOLO chạy mỗi N frame (chỉ dùng khi mất mặt → giãn không hại độ nhạy) |
| `HEADLESS` | `1` | `1`=không mở cửa sổ (Pi chạy không màn hình); `0` khi dev có HDMI |
| `DETECTION_MODE` | `multi_signal` | hoặc `strict` (safety-first, bỏ multi-signal, tiết kiệm 6–12ms/frame) |
| `OCCLUSION_THRESHOLD_SEC` | `15` | Số giây bị che liên tục → alert |
| `CALIBRATION_SEC` | `5` | Thời gian calibrate baseline lúc khởi động |
| `CONFIRM_FRAMES` | `10` | Tăng (15–20) → ổn định hơn, response chậm hơn chút |
| `SMOOTHER_MAX_MISS` | `3` | Tăng `4` nếu landmark nhảy nhiều |
| `AUTO_RECAL_AFTER_SEC` | `1800` | Tự recalib sau 30 phút safe liên tục (0=tắt) |
| `MP_DETECTION_CONFIDENCE` | `0.6` | MediaPipe min detection conf (chỉ `multi_signal`) |
| `MP_TRACKING_CONFIDENCE` | `0.6` | MediaPipe min tracking conf |
| `BLUR_DROP_FRAC` | `0.45` | Blur-gate: độ nét < frac×baseline → bỏ phiếu edge/lap (chống false alert do mờ) |
| `HEALTH_MIN_FPS` | `3.0` | Watchdog: FPS dưới mức này (kéo dài) → cảnh báo suy giảm |
| `HEALTH_DARK_LUMA` | `25` | Watchdog: độ sáng TB dưới mức này → cảnh báo "quá tối" |
| `HEALTH_DEGRADE_SEC` | `20` | Sự cố kéo dài ≥ N giây mới cảnh báo (chống spam) |
| `HEARTBEAT_SEC` | `0` (tắt) | Gửi Telegram "vẫn đang canh" mỗi N giây (vd `21600`=6h) |
| `STARTUP_NOTIFY` | `1` (bật) | Lúc khởi động gửi Telegram yêu cầu đưa mặt trẻ vào khung + xác nhận khi hiệu chỉnh xong |
| `CALIB_REMIND_SEC` | `60` | Nhắc lại "đưa mặt vào khung" mỗi N giây nếu mãi chưa hiệu chỉnh được |

> 🚫 **Không có `YOLO_DEVICE`.** Project CPU-only, không hỗ trợ NVIDIA GPU. Pi 4 cũng không có NPU.
> ⚠️ Đừng commit `.env` lên git (chứa token). Thêm `.env` vào `.gitignore`.

---

## 8. Verify cài đặt

Chạy theo thứ tự, mỗi bước phải pass.

### 8.1 Test logic (không cần camera)
```bash
cd ~/baby-ai-alert
source venv/bin/activate

python tests/test_state_machine.py        # Mong đợi: 🎉 14/14 test PASS
python tests/test_occlusion_detector.py    # Mong đợi: 🎉 16/16 test PASS
python tests/test_scene_monitor.py         # Mong đợi: 🎉 7/7 test PASS
python tests/test_alert_policy.py          # Mong đợi: 🎉 14/14 test PASS
```

### 8.2 Scan camera
```bash
python scripts/test_webcam.py --scan
```

### 8.3 Verify FPS thực tế (headless, không HDMI)
```bash
python scripts/test_webcam.py --source 0 --headless --save-snapshot events/webcam_test.jpg
```
Mở `events/webcam_test.jpg` kiểm tra hình rõ. FPS thực nên ≥15 ở 640×480. Nếu thấp:
- Cắm cổng USB 3.0 (xanh)
- Cáp USB ngắn, tốt
- Đảm bảo `.env` đã set 640×480

### 8.4 Benchmark pipeline (đo đúng cấu hình production Pi 4)
```bash
CAMERA_WIDTH=640 CAMERA_HEIGHT=480 python scripts/benchmark.py --frames 100
```
Kỳ vọng trên Pi 4 (CPU, 480p): end-to-end **~120–180ms (~6–8 FPS)**. Miễn **≥6 FPS** là state machine + smoother chạy đúng. Nếu < 6 FPS → xem [§11](#11-tối-ưu-hiệu-năng-cho-pi-4).

---

## 9. Chạy chương trình

### 9.1 Chạy chương trình

Default trong code **đã đúng cho Pi 4** (640×480 @15fps), nên chạy thẳng cũng ra đúng cấu hình:
```bash
cd ~/baby-ai-alert
source venv/bin/activate
python src/main.py
```

Nhưng để **tự nạp token** trong `.env` (§7), nên dùng wrapper `run.sh` — Python không tự đọc `.env`, phải `source` vào shell trước:

```bash
cat > ~/baby-ai-alert/run.sh << 'EOF'
#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -f .env ]; then
    set -a; source .env; set +a    # nạp token (+ override nếu có) vào môi trường
fi
source venv/bin/activate
exec python src/main.py
EOF
chmod +x ~/baby-ai-alert/run.sh

./run.sh
```

> 💡 Chạy thẳng `python src/main.py` (không `run.sh`) thì camera/FPS vẫn đúng (640×480 — đã là default), chỉ **không nạp token trong `.env`** → app dùng token demo. Để gửi alert về Telegram của bạn thì phải `run.sh`/systemd, hoặc tự `export TELEGRAM_TOKEN=...` trước khi chạy.

#### Verify cấu hình lúc khởi động

Log khởi động in dòng `Camera`. Trên Pi 4 phải thấy **640x480**:
```
   Camera           : 0 → 640x480 @ 15fps      ✅ ĐÚNG (default Pi 4)
```
Nếu thấy `1280x720` → có ai đó set `CAMERA_WIDTH/HEIGHT` cao trong `.env`. Pi 4 không kham nổi 720p (FPS tụt còn ~4–6 → ảnh mờ → mũi/miệng mất chi tiết → **báo nhầm**). Hạ lại 640×480.

### 9.2 Test 4 kịch bản cảnh báo

| # | Kịch bản | Kỳ vọng |
|---|---|---|
| 1 | **Mặt sạch ngồi yên 30s** | votes 0/4 hoặc 1/4. KHÔNG alert. |
| 2 | **Tay che mũi/miệng 15s+** | votes 2–3/4 (`lap` giảm rõ). Sau 15s → Telegram alert `MUI/MIENG BI CHE` |
| 3 | **Chăn phủ kín mặt 15s+** | votes 4/4 ngay. Sau 15s → alert (hoặc `Mất hoàn toàn khuôn mặt`) |
| 4a | **Chưa đặt trẻ vào (khung trống) + YOLO không thấy người** | state `NO_FACE`, KHÔNG gửi tin |
| 4b | **Rời khung NGẮN (<15s) rồi quay lại** | Chưa đủ ngưỡng → KHÔNG gửi tin, về lại `SAFE` |
| 4c | **Đã thấy mặt rồi rời khung ≥15s** (kể cả YOLO không thấy người) | ⚠️ **VẪN gửi** `Mất hoàn toàn khuôn mặt` — chủ đích **safety-first** (xem ghi chú dưới) |

> ⚠️ **Quan trọng — thiết kế safety-first:** một khi đã thấy mặt trẻ rồi mặt **biến mất**, hệ thống coi đó là **nghi bị phủ kín** và đếm tới 15s rồi báo, **bất kể YOLO nói gì**. Lý do: camera đặt **top-down** xuống cũi → YOLO (train trên người đứng) hay **sót** trẻ nằm bị chăn phủ → nếu tin "YOLO không thấy người = rời khung" sẽ **bỏ lọt ca bị che thật**. Hệ thống chấp nhận **báo nhầm khi cha mẹ bế trẻ đi** để TUYỆT ĐỐI không miss. YOLO chỉ dùng để: (a) **bắt đầu đếm** khi thấy người mà không thấy mặt (chưa từng track mặt), và (b) **im lặng** khi khung trống chưa từng thấy mặt (4a). → Khi demo "rời khung không báo", hãy quay lại **trước 15s** (4b).

Mỗi event lưu `events/face_covered_<timestamp>.jpg` + `.json` (kèm 4 signal values) ở lần cảnh báo ĐẦU. Khi vẫn còn che → re-alert Telegram mỗi 15s (kèm ảnh mới) nhưng KHÔNG ghi thêm file trùng (tránh phình đĩa).

### 9.3 Recalibration
- **GUI mode** (có HDMI, `HEADLESS=0`): nhấn phím `R`.
- **Headless** (systemd): gửi SIGUSR1:
  ```bash
  pgrep -f "src/main.py"     # tìm PID
  kill -USR1 <PID>            # trigger recalibrate
  ```
- **Auto**: tự recal sau `AUTO_RECAL_AFTER_SEC` giây safe liên tục (default 30 phút).

### 9.4 Xem live view khi debug
```bash
ssh -X pi@babypi.local
HEADLESS=0 python src/main.py     # cần X-forwarding hoặc HDMI gắn trực tiếp
```

---

## 10. Autostart bằng systemd

> 🎯 Mục tiêu: cắm điện → Pi boot → app **tự chạy**. Mất điện/reboot → tự chạy lại. Crash → tự khởi động lại sau 10s.

### 10.1 Cài 1 lệnh (KHUYẾN NGHỊ)

Script tự dò user/group/python/path (không hardcode đường dẫn):
```bash
sudo bash scripts/setup_autostart.sh
```
Nó tự: sinh `/etc/systemd/system/baby-monitor.service` từ template (điền đúng user/path/python của bạn), tạo file log + logrotate 7 ngày, cron dọn `events/` > 30 ngày, thêm user vào group `video`, rồi enable + start.

Service template ([deploy/baby-monitor.service](deploy/baby-monitor.service)) đã vá sẵn cho chạy 24/7:
- `After=network-online.target` → chờ mạng THẬT SỰ lên (WiFi+DHCP+DNS) → **alert Telegram đầu sau boot không bị trượt vì mạng chưa sẵn**
- `ExecStartPre` chờ `/dev/video0` tối đa 30s (USB webcam chưa sẵn ngay lúc boot)
- `Restart=always` + `RestartSec=10` + `MemoryMax=1G`
- `EnvironmentFile=-/.../.env` → **tự nạp file `.env` Pi 4 ở §7** khi boot

> ⚠️ Để `network-online.target` có tác dụng (Bookworm dùng NetworkManager), bật dịch vụ chờ mạng:
> ```bash
> sudo systemctl enable NetworkManager-wait-online.service
> ```
> App còn có **warm-up + retry/backoff** khi gửi Telegram ([src/main.py](src/main.py)): lúc khởi động tự "hâm" kết nối ở thread nền, và mỗi alert thử lại nhiều lần (0→2→5→10→20s) nếu mạng cold — nên kể cả mạng lên trễ, alert đầu vẫn tới thay vì bị bỏ.

### 10.2 Kiểm tra & vận hành
```bash
sudo systemctl status baby-monitor
journalctl -u baby-monitor -f       # log live
tail -f /var/log/baby-monitor.log

sudo systemctl stop baby-monitor    # tạm dừng để dev (giải phóng camera)
sudo systemctl start baby-monitor

sudo bash scripts/setup_autostart.sh --uninstall   # gỡ tự chạy
```

Chạy **1 lần duy nhất** — từ đó mỗi lần boot tự chạy, không cần gõ gì nữa.

---

## 11. Tối ưu hiệu năng cho Pi 4

Nếu [§8.4](#84-benchmark-pipeline-đo-đúng-cấu-hình-production-pi-4) cho < 6 FPS, làm lần lượt cho tới khi đủ:

1. **Hạ resolution thêm** (giảm MediaPipe ~50% mỗi nấc):
   ```ini
   CAMERA_WIDTH=480
   CAMERA_HEIGHT=360
   ```
2. **Giãn YOLO**: `YOLO_EVERY=15` (YOLO chỉ phân biệt "rời khung" vs "bị phủ", giãn không hại độ nhạy che).
3. **Đổi sang `strict` mode**: `DETECTION_MODE=strict` — bỏ tính multi-signal, tiết kiệm 6–12ms/frame (đánh đổi: nhiều false positive hơn, nhưng safety-first).
4. **Tăng `CONFIRM_FRAMES`** nếu votes dao động vì FPS thấp (ổn định hơn, response chậm hơn chút).

### 11.1 Kiểm tra thermal throttle (rất quan trọng trên Pi 4)
```bash
vcgencmd measure_temp           # nhiệt độ; > 80°C → cần quạt/heatsink tốt hơn
vcgencmd get_throttled          # 0x0 = OK. Khác 0 = đang/đã throttle hoặc undervoltage
```
Giải mã `get_throttled` (bit thường gặp):
- `0x50000` / bit 16,18 = **đã từng** throttle/undervoltage (lịch sử)
- `0x50005` = **đang** undervoltage + throttle → **đổi nguồn 5V/3A chính hãng**

### 11.2 (Optional) Ép xung nhẹ
Nếu có tản nhiệt tốt, thêm vào `/boot/firmware/config.txt` (Bookworm; bản cũ là `/boot/config.txt`):
```ini
over_voltage=4
arm_freq=1900
```
→ `sudo reboot`. ⚠️ Chỉ làm khi tản nhiệt đủ; ép xung mà nóng sẽ throttle ngược lại.

### 11.3 Tính năng chất lượng bổ sung (mới)

Ba lớp logic nâng chất lượng/độ an toàn, đều rẻ CPU (~5ms/frame, dùng chung 1 ảnh xám downscale trong [src/scene_monitor.py](src/scene_monitor.py)):

1. **Blur-gate (BẬT sẵn)** — khi CẢ khung đột ngột mờ (autofocus hunting / motion blur ở FPS thấp), `edge`/`lap` tụt về 0 không phải do bị che → detector **bỏ 2 phiếu texture** vòng đó, chỉ tin hist+skin. Diệt đúng lớp false-positive đã gặp. Vật/tay che thật chỉ làm mờ vùng mặt (nền vẫn nét) → độ nét toàn cục không sụt → gate **không** kích → vẫn phát hiện che bình thường. Chỉnh `BLUR_DROP_FRAC`.

2. **Watchdog + heartbeat (BẬT sẵn, trừ heartbeat)** — tự giám sát để **không fail âm thầm**: camera ĐƠ (frame trùng) / phòng QUÁ TỐI / FPS quá thấp kéo dài ≥ `HEALTH_DEGRADE_SEC` → gửi `⚠️ GIÁM SÁT SUY GIẢM`, và báo `✅ Đã khôi phục` khi hết. Bật heartbeat định kỳ ("vẫn đang canh") bằng `HEARTBEAT_SEC` (vd 6h).

3. **Thông báo khởi động / hiệu chỉnh (BẬT sẵn)** — lúc khởi động (kể cả autostart lúc boot) gửi Telegram **🟢 yêu cầu đưa mặt trẻ vào khung** để hiệu chỉnh; **nhắc lại** mỗi `CALIB_REMIND_SEC` nếu mãi chưa thấy mặt (vd quên chỉnh camera); và **✅ xác nhận "bắt đầu giám sát"** khi hiệu chỉnh xong (kèm cảnh báo nếu chất lượng thấp). → Người dùng không nhận được tin xác nhận = biết ngay hệ thống **chưa** giám sát. Tắt bằng `STARTUP_NOTIFY=0`.
   > 🔍 **Cổng chất lượng hiệu chỉnh:** baseline là "định nghĩa mặt sạch" mà mọi lần phát hiện che về sau so sánh với — học từ frame **tối/mờ** sẽ ra baseline rác, kém tin cậy ngoài thực địa. Vì vậy pha hiệu chỉnh **chỉ gom mẫu từ frame ĐẠT** (độ sáng ≥ `HEALTH_DARK_LUMA`, không bị blur-gate). Nếu điều kiện kém kéo dài ≥ `CALIB_COND_GRACE_SEC` (mặc định 4s), hệ thống gửi Telegram **hướng dẫn cụ thể** ("phòng quá tối → bật đèn" / "hình mờ → chỉnh tiêu cự") và tiếp tục chờ điều kiện tốt thay vì học baseline kém.

---

## 12. (Optional) Kích relay khi có cảnh báo

Pi 4 dùng thư viện GPIO **`RPi.GPIO`** (hoặc `lgpio` trên Bookworm). Layout 40-pin chuẩn Raspberry Pi.

### 12.1 Đấu dây module relay 4 kênh
| Relay pin | Pi 4 GPIO (BCM) | Pi 4 physical pin |
|---|---|---|
| VCC | 5V | Pin 2 |
| GND | GND | Pin 6 |
| IN1 | GPIO 17 | Pin 11 |
| IN2 | GPIO 27 | Pin 13 |
| IN3 | GPIO 22 | Pin 15 |
| IN4 | GPIO 23 | Pin 16 |

> ⚠️ GPIO Pi 4 xuất 3.3V — một số relay 5V không trigger nổi. Dùng relay loại 3.3V-trigger hoặc thêm transistor đệm.

### 12.2 Cài thư viện GPIO (Bookworm)
```bash
source ~/baby-ai-alert/venv/bin/activate
pip install lgpio          # khuyến nghị trên Bookworm
# hoặc giữ API cũ: pip install rpi-lgpio   (shim RPi.GPIO chạy trên lgpio)
```

### 12.3 Wire vào code
Trong `_dispatch_alert()` của [src/main.py](src/main.py), thêm trigger khi `should_alert`:
```python
import lgpio, threading
_h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(_h, 17)

def _trigger_relay(channel=17, duration_sec=5):
    lgpio.gpio_write(_h, channel, 1)
    threading.Timer(duration_sec, lambda: lgpio.gpio_write(_h, channel, 0)).start()
```
Gọi `_trigger_relay()` trong nhánh `if result.should_alert`. Ping mình nếu cần wire hoàn chỉnh.

---

## 13. (Optional) Tăng tốc YOLO bằng Google Coral USB TPU

Pi 4 **không có NPU**. Muốn tăng tốc YOLO (chỉ cần khi benchmark quá chậm) → gắn **Google Coral USB Accelerator** (Edge TPU, 4 TOPS):

1. Convert `yolov8n.pt` → TFLite int8 → Edge TPU `.tflite` (làm trên máy dev, dùng `edgetpu_compiler`).
2. Trên Pi cài `pycoral` + `libedgetpu` (cắm Coral vào **USB 3.0**).
3. Viết wrapper `CoralPersonDetector` thay `PersonDetector` trong [src/main.py](src/main.py) — cùng interface `has_person()`, chỉ đổi inference engine.

> Đây là việc riêng ngoài scope hiện tại. Với mục đích phát hiện che mũi/miệng, YOLO chạy thưa (`YOLO_EVERY=10`) trên CPU thường là **đủ** — Coral chỉ cần nếu bạn ép real-time cao. Ping mình khi cần làm.

---

## 14. Troubleshooting

### `uname -m` ra `armv7l` (không phải `aarch64`)
Đang chạy OS 32-bit → MediaPipe không cài được. **Flash lại Raspberry Pi OS (64-bit)** ([§2](#2-cài-hệ-điều-hành-raspberry-pi-os-64-bit)).

### `error: externally-managed-environment` khi `pip install`
Bookworm chặn cài vào Python hệ thống (PEP 668). **Phải dùng venv**: `python3 -m venv venv && source venv/bin/activate` rồi mới `pip install`. (Đừng dùng `--break-system-packages`.)

### `Could not find a version that satisfies the requirement mediapipe`
- OS 32-bit (xem trên), hoặc Python 3.12+ chưa có wheel → cài Python 3.11 ([§5.2](#52-verify-python-311-bookworm-có-sẵn)).
- pip cũ → `pip install --upgrade pip`.

### `Killed` khi `pip install` (OOM)
RAM hết → tạo swap 2GB ([§3.4](#34-tạo-swap-bắt-buộc-nếu-ram--4gb)). Hoặc cài từng package một.

### pip kéo về `nvidia_cudnn_cu13` (433MB) / `cuda_toolkit`
torch≥2.4 declare nvidia làm hard dep dù CPU-only. Fix:
```bash
git pull                                   # lấy requirements.txt pin torch<2.4
bash scripts/fix_env.sh                     # uninstall torch+nvidia, cài lại đúng
```

### `NumPy 2.x cannot be run...` / opencv yêu cầu numpy≥2
ABI mismatch. Fix triệt để:
```bash
bash scripts/fix_env.sh
```

### App in `❌ OpenCV variant ...` / `❌ Phát hiện nhiều opencv variant` rồi exit
Guard tầng 2/3 fire → `bash scripts/fix_env.sh`.

### `can't open camera by index` / `[WARN] cap_v4l ...`
- User chưa thuộc group video → `sudo usermod -aG video $USER` + **logout/login**.
- Camera bị app khác giữ → `sudo fuser -k /dev/video0` (hoặc `sudo systemctl stop baby-monitor`).
- Sai index → `v4l2-ctl --list-devices`.

### FPS thực < 6 (state machine không đủ ổn định)
- USB 2.0 thay vì 3.0 → đổi cổng xanh.
- Resolution quá cao → hạ xuống 480×360 ([§11](#11-tối-ưu-hiệu-năng-cho-pi-4)).
- **Thermal throttle**: `vcgencmd measure_temp` > 80°C → gắn quạt; `vcgencmd get_throttled` ≠ 0x0 → **đổi nguồn 5V/3A**.

### Biểu tượng tia chớp ⚡ / undervoltage / USB chập chờn
Nguồn yếu — dùng **adapter USB-C 5V/3A chính hãng**, không dùng cáp sạc điện thoại.

### Tay che mũi/miệng NHƯNG không alert (false negative)
- Xem debug bar dòng `MOUTH`: khi tay che, `lap=` phải < 50 → vote. Nếu vẫn > 50: tăng `LAPVAR_ABSOLUTE_FLOOR` trong [src/occlusion_detector.py](src/occlusion_detector.py) (vd 80).
- `edge` của tay > 0.020 → tăng `EDGE_ABSOLUTE_FLOOR` (vd 0.030).
- votes dao động → giảm `CONFIRM_FRAMES` (7–8), tăng `SMOOTHER_MAX_MISS` (4–5).
- Recalibrate (`R` hoặc `kill -USR1`).

### Mặt sạch nhưng vẫn alert (false positive)
- **NGUYÊN NHÂN SỐ 1 TRÊN PI 4: camera đang chạy ở 720p.** Kiểm tra log khởi động có in `640x480` không ([§9.1](#91-chạy-chương-trình)). Nếu in `1280x720` → có ai set `CAMERA_WIDTH/HEIGHT` cao trong `.env`/env → Pi 4 tụt FPS → ảnh mờ → `edge≈0`/`lap≈0` ở mũi/miệng → báo nhầm. Fix: bỏ override, để default 640×480.
- `quality` < 0.5 → calibration kém → recalibrate trong điều kiện ổn định.
- Tăng `CONFIRM_FRAMES` (15–20) để vài frame nhòe không kịp confirm. Nếu riêng cái mũi hay vote (`edge=0.000 lap` thấp dù da rõ), nới `EDGE_DROP_FRAC`/`LAPVAR_DROP_FRAC` trong [src/occlusion_detector.py](src/occlusion_detector.py) (vd 0.55 / 0.70) để texture phải tụt nhiều hơn mới vote.

### `Failed to send Telegram message`
- Không internet → `ping 8.8.8.8`. Token/chat ID sai → test curl ([§6.3](#63-test-gửi-từ-chính-pi-4)).

### systemd service không start
```bash
sudo systemctl status baby-monitor
journalctl -u baby-monitor -n 50
```
Thường vì: `.env` sai path/permission; venv path sai; owner project khác user → `sudo chown -R $USER:$USER ~/baby-ai-alert`.

### `cannot connect to X server`
Chạy headless mà code gọi `cv2.imshow()` → đảm bảo `HEADLESS=1` trong `.env`.

### Chạy ổn vài giờ rồi crash
- Thẻ SD bad block → `dmesg | grep -i error`. Production nên dùng SSD USB ([§2.5](#25-khuyến-nghị-production-boot-từ-ssd-usb)).
- `MemoryMax=1G` trong systemd sẽ tự restart nếu rò rỉ.
- Nhiệt độ → gắn quạt.

### MediaPipe load chậm (10–20s) lần đầu khởi động
Bình thường trên ARM lần đầu (extract model). Lần boot sau nhanh hơn (~3–5s).

---

## 15. Tổng kết stack

```
Raspberry Pi 4 Model B (BCM2711, 4GB/8GB RAM)
├── Raspberry Pi OS 64-bit (Bookworm)  ← BẮT BUỘC 64-bit
├── Logitech USB webcam (cổng USB 3.0)
├── Heatsink + quạt (bắt buộc, tránh throttle)
├── Nguồn USB-C 5V/3A chính hãng
├── Module relay 4 kênh → GPIO (RPi.GPIO/lgpio) — optional, §12
│
├── ~/baby-ai-alert/
│   ├── venv/                     ← Python 3.11 isolated env (PEP 668)
│   ├── requirements.txt          ← deps pinned (numpy<2, opencv<4.11, torch<2.4)
│   ├── src/
│   │   ├── main.py               ← entrypoint, 3 guard env, signal handlers
│   │   ├── state_machine.py      ← FSM phát hiện che mũi/miệng
│   │   └── occlusion_detector.py ← Multi-signal voting (hist+skin+edge+lap_var)
│   ├── scripts/
│   │   ├── install_pi4.sh       ← One-shot install cho Pi 4 (ARM64 CPU-only)
│   │   ├── fix_env.sh            ← Fix opencv/numpy/nvidia variants
│   │   ├── setup_autostart.sh    ← Cài systemd autostart (tự dò user/path)
│   │   ├── test_webcam.py        ← Verify camera + FPS
│   │   └── benchmark.py          ← Đo pipeline
│   ├── tests/                    ← 11 + 15 test
│   ├── .env                      ← token Telegram (camera/FPS đã là default code)
│   ├── .env.example             ← mẫu .env (chủ yếu cho token)
│   ├── run.sh                    ← wrapper nạp .env rồi chạy
│   ├── events/                   ← snapshot + JSON khi alert (cron dọn 30 ngày)
│   └── yolov8n.pt                ← person detection (CPU; hoặc Coral .tflite, §13)
│
└── systemd:
    ├── baby-monitor.service      ← autostart + restart + nạp .env + chờ camera
    ├── /etc/logrotate.d/...      ← rotate log 7 ngày
    └── cron: dọn events/ 02:00 mỗi ngày
```

### Các điểm cần nhớ khi vận hành trên Pi 4

| Hạng mục | Giá trị / lưu ý |
|---|---|
| OS | **Raspberry Pi OS 64-bit (Bookworm)** — BẮT BUỘC 64-bit (MediaPipe không có wheel 32-bit) |
| Camera/FPS | **640×480 @15fps** (default code). Đừng lên 720p → Pi 4 tụt FPS → báo nhầm |
| `.env` | Chỉ cần để đặt **token Telegram** thật; phần hiệu năng đã là default |
| Tăng tốc YOLO | Pi 4 không có NPU → CPU; muốn nhanh hơn gắn **Coral USB TPU** (§13) |
| GPIO relay | `RPi.GPIO` / `lgpio` (§12) |
| pip vào system python | **bị chặn (PEP 668) → bắt buộc venv** |
| Nguồn/nhiệt | USB-C 5V/3A chính hãng + heatsink/quạt (tránh undervoltage + throttle) |

---

## Báo lỗi

```bash
cat /proc/device-tree/model > diag.txt
uname -m >> diag.txt
free -h >> diag.txt
vcgencmd measure_temp >> diag.txt
vcgencmd get_throttled >> diag.txt
v4l2-ctl --list-devices >> diag.txt
python --version >> diag.txt
pip list >> diag.txt
journalctl -u baby-monitor -n 100 >> diag.txt
CAMERA_WIDTH=640 CAMERA_HEIGHT=480 python scripts/benchmark.py --frames 50 >> diag.txt 2>&1
```
Gửi `diag.txt` + mô tả tình huống.
