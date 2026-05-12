# Hướng dẫn cài đặt Baby AI Alert trên Orange Pi 5

Phiên bản cài đặt cho board nhúng **Orange Pi 5 / 5B / 5 Plus** (RK3588 SoC, Linux ARM64).

## Mục lục

1. [Phần cứng đang dùng](#1-phần-cứng-đang-dùng)
2. [Cài hệ điều hành](#2-cài-hệ-điều-hành)
3. [Setup ban đầu](#3-setup-ban-đầu-ssh--update--swap)
4. [Cài system dependencies](#4-cài-system-dependencies)
5. [Cài Python + project dependencies](#5-cài-python--project-dependencies)
6. [Tạo Telegram bot + lấy token](#6-tạo-telegram-bot--lấy-token)
7. [Cấu hình bằng `.env`](#7-cấu-hình-bằng-env)
8. [Verify cài đặt](#8-verify-cài-đặt)
9. [Chạy chương trình](#9-chạy-chương-trình)
10. [Autostart bằng systemd](#10-autostart-bằng-systemd)
11. [(Optional) Tăng tốc bằng NPU RK3588](#11-optional-tăng-tốc-bằng-npu-rk3588)
12. [(Optional) Kích relay khi có cảnh báo](#12-optional-kích-relay-khi-có-cảnh-báo)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Phần cứng đang dùng

| Thành phần | Specs |
|---|---|
| **Board** | Orange Pi 5 / 5B / 5 Plus (RK3588, 8 core ARM Cortex-A76+A55, NPU 6 TOPS) |
| **RAM** | 4 GB / 8 GB / 16 GB |
| **Storage** | microSD ≥32GB (Class 10) hoặc eMMC ≥32GB hoặc NVMe SSD |
| **Camera** | Logitech USB webcam (C270/C310/...) |
| **Mạng** | Ethernet hoặc WiFi |
| **Tản nhiệt** | RK3588 nóng — **bắt buộc** có heatsink + quạt, không có sẽ thermal throttle |
| **Nguồn** | USB-C 5V/4A (≥20W) — dùng adapter chuẩn, đừng cắm cable điện thoại 5V/2A |

### Cách verify model chính xác (sau khi cài OS):
```bash
cat /proc/device-tree/model
# Output ví dụ: "Xunlong Orange Pi 5"
```

---

## 2. Cài hệ điều hành

Khuyến nghị **Ubuntu 22.04 ARM64** (chính thức từ Orange Pi) hoặc **Armbian**.

### 2.1 Tải image
- Ubuntu chính thức: http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-5.html
- Armbian: https://www.armbian.com/orange-pi-5/

Chọn **Ubuntu 22.04 Server** (không cần desktop nếu chạy headless — đỡ tốn RAM).

### 2.2 Flash vào SD card

Dùng **Balena Etcher** (Windows/Mac) hoặc `dd` (Linux):
```bash
# Linux
sudo dd if=Orangepi5_x.x.x_ubuntu_jammy_server_linux.img of=/dev/sdX bs=4M status=progress
sync
```

### 2.3 Boot lần đầu

1. Cắm SD card, cắm HDMI + bàn phím + nguồn USB-C
2. Đợi 1-2 phút boot xong
3. Login mặc định: `root` / `orangepi` (đổi password ngay sau khi login)
4. Tạo user thường, KHÔNG chạy app dưới root:
   ```bash
   adduser pi
   usermod -aG sudo,video pi
   su - pi
   ```

### 2.4 Boot từ eMMC / NVMe (khuyến nghị production)

SD card hay hỏng sau vài tháng vận hành 24/7. Nếu board có eMMC hoặc gắn được NVMe → flash OS vào đó:

```bash
sudo orangepi-config   # menu: System → Install to eMMC/NVMe
```

---

## 3. Setup ban đầu (SSH + update + swap)

### 3.1 Mạng + SSH

```bash
# Kiểm tra IP
ip a

# Bật SSH (thường đã bật sẵn)
sudo systemctl enable --now ssh
```

Từ máy tính khác:
```bash
ssh pi@<orange-pi-ip>
```

### 3.2 Update hệ thống

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

### 3.3 Đồng bộ giờ NTP

Timestamp trong alert Telegram phải đúng:
```bash
sudo timedatectl set-timezone Asia/Ho_Chi_Minh
sudo timedatectl set-ntp true
timedatectl status   # verify
```

### 3.4 Tạo swap (board RAM nhỏ)

Nếu RAM ≤ 4GB → tạo swap 2GB để `pip install mediapipe` không bị OOM lúc build:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h   # verify
```

---

## 4. Cài system dependencies

### 4.1 Build tools + libraries (cho việc compile pip wheels nếu cần)

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

### 4.2 Quyền truy cập camera

User chạy app phải thuộc group `video`:
```bash
sudo usermod -aG video $USER
# Logout/login lại để áp dụng
```

### 4.3 Kiểm tra webcam được nhận

Cắm Logitech webcam vào cổng USB **3.0** (cổng xanh), không cắm USB 2.0 (sẽ giới hạn FPS):

```bash
v4l2-ctl --list-devices
# Phải thấy: HD Webcam C270 hoặc tương tự
# /dev/video0, /dev/video1...

v4l2-ctl -d /dev/video0 --list-formats-ext
# Liệt kê resolution + FPS hỗ trợ
```

---

## 5. Cài Python + project dependencies

### 5.1 Clone source code

```bash
cd /opt
sudo git clone <your-repo-url> baby-monitor
sudo chown -R $USER:$USER baby-monitor
cd baby-monitor
```

(Hoặc copy thư mục `baby-ai-alert/` từ máy dev qua bằng `scp`/`rsync`.)

### 5.2 Tạo virtualenv

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

### 5.3 Cài MediaPipe ARM64 — phần khó nhất

MediaPipe có wheel cho `aarch64` từ phiên bản 0.10.x trở đi nhưng chỉ hỗ trợ Python **3.9 - 3.11**. Kiểm tra Python version:

```bash
python --version
```

- Nếu là 3.10 hoặc 3.11 → cài bình thường:
  ```bash
  pip install mediapipe
  ```
- Nếu Python 3.12+ → cài Python 3.11 song song:
  ```bash
  sudo apt install -y python3.11 python3.11-venv python3.11-dev
  python3.11 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install mediapipe
  ```

### 5.4 Cài deps từ `requirements.txt`

```bash
pip install -r requirements.txt
```

`requirements.txt` đã pin sẵn:
- `numpy>=1.24,<2`  ← mediapipe 0.10.x compile chống NumPy 1.x
- `opencv-python>=4.8,<4.11`  ← opencv ≥4.11 compile chống NumPy 2.x
- `mediapipe>=0.10`, `python-telegram-bot>=20`, `ultralytics>=8.0`

> **Lưu ý**: `ultralytics` kéo PyTorch. Trên ARM64, pip sẽ cài bản **CPU-only** (~200MB) — đúng rồi, RK3588 không có CUDA. Tăng tốc qua NPU làm ở mục 11.

### 5.5 Verify + fix version conflicts

```bash
bash scripts/fix_env.sh
```

Script này tự động:
- Uninstall **tất cả** opencv variant (`opencv-python`, `opencv-contrib-python`, `opencv-python-headless`, `opencv-contrib-python-headless`) để clean slate
- Cài lại `numpy<2` + `opencv-python<4.11` để đảm bảo binary tương thích
- In version cuối cùng để verify

Sau đó verify import:
```bash
python -c "import cv2, mediapipe, numpy, telegram, ultralytics; print('All imports OK')"
```

Phải in `All imports OK`. Nếu fail → xem mục 13 (Troubleshooting).

### 5.6 (Tự động) App có 3 lớp guard chặn env sai

`src/main.py` tự refuse to start nếu phát hiện:
1. `numpy >= 2` (mediapipe sẽ crash)
2. Bất kỳ `opencv-python(-contrib)(-headless) >= 4.11` (mismatch numpy 1.x)
3. Nhiều opencv variant cài cùng lúc (xung đột cv2 namespace)

Khi guard fire, app in command fix rõ ràng và exit. Không có khả năng silent corruption.

---

## 6. Tạo Telegram bot + lấy token

### 6.1 Tạo bot
1. Mở Telegram, tìm `@BotFather`
2. Gõ `/newbot`, đặt tên + username (kết thúc bằng `bot`)
3. Copy **token** dạng `1234567890:ABC...`

### 6.2 Lấy chat ID
1. Gõ vài tin nhắn cho bot vừa tạo
2. Truy cập:
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
3. Tìm `"chat":{"id":7316578932,...}` → đó là **chat ID**

### 6.3 Test gửi tin từ Orange Pi

```bash
TOKEN="..."
CHAT="..."
curl -s "https://api.telegram.org/bot$TOKEN/sendMessage?chat_id=$CHAT&text=test+from+opi"
```

Phải nhận tin "test from opi" trong Telegram.

---

## 7. Cấu hình bằng `.env`

```bash
cat > /opt/baby-monitor/.env << 'EOF'
# ---- Bắt buộc ----
TELEGRAM_TOKEN=1234567890:ABC...
TELEGRAM_CHAT_ID=7316578932

# ---- Camera ----
CAMERA_SOURCE=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30

# ---- Headless (bắt buộc cho systemd, không có HDMI) ----
HEADLESS=1

# ---- Detection logic ----
OCCLUSION_THRESHOLD_SEC=15   # bao nhiêu giây bị che thì alert
CALIBRATION_SEC=5            # thời gian calibrate ban đầu
CONFIRM_FRAMES=10            # frame confirm bị che (tăng = ổn hơn, chậm hơn)
SMOOTHER_MAX_MISS=3          # cho phép N frame miss trong window confirm
MIN_QUALITY_WARN=0.5         # quality < này → warn UI gợi ý recalib
AUTO_RECAL_AFTER_SEC=1800    # auto-recalibrate sau N giây safe liên tục (0=tắt)

# ---- YOLO ----
YOLO_DEVICE=cpu              # cpu | cuda (chỉ cuda sau khi setup RKNN)
YOLO_EVERY=5                 # chạy YOLO mỗi N frame để đỡ tải

PYTHONIOENCODING=utf-8
EOF

chmod 600 /opt/baby-monitor/.env   # bảo vệ token
```

| Biến | Default | Tác dụng |
|---|---|---|
| `CAMERA_SOURCE` | `0` | `0`/`/dev/video0`/`rtsp://...` |
| `CAMERA_WIDTH/HEIGHT/FPS` | 1280/720/30 | Giảm xuống 640/480/15 nếu OPi yếu |
| `HEADLESS` | `0` | **Phải set `1` trên systemd** (không có X server) |
| `OCCLUSION_THRESHOLD_SEC` | `15` | Ngưỡng giây để alert |
| `CONFIRM_FRAMES` | `10` | Tăng → ổn định hơn, chậm hơn 0.3s |
| `SMOOTHER_MAX_MISS` | `3` | Cho phép landmark jumpy 3 frame không reset đếm |
| `AUTO_RECAL_AFTER_SEC` | `1800` | Auto-recalib sau 30 phút safe liên tục |
| `YOLO_DEVICE` | `cpu` | `cuda` sau khi setup RKNN |
| `YOLO_EVERY` | `5` | YOLO chạy mỗi 5 frame |

> ⚠️ Đừng commit `.env` lên git. Thêm `.env` vào `.gitignore`.

---

## 8. Verify cài đặt

Chạy theo thứ tự, mỗi bước phải pass.

### 8.1 Test logic (không cần camera)
```bash
cd /opt/baby-monitor
source venv/bin/activate

# State machine logic (FSM phát hiện ngạt thở)
python tests/test_state_machine.py
# Kỳ vọng: 🎉 11/11 test PASS

# Multi-signal occlusion detector (hist + skin + edge + lap_var)
python tests/test_occlusion_detector.py
# Kỳ vọng: 🎉 14/14 test PASS
```

Trong test_occlusion_detector có các test quan trọng:
- `test_check_on_hand_alerts` — verify tay che mặt phải alert
- `test_hand_stability_under_landmark_drift` — verify ổn định khi landmark nhảy
- `test_check_on_blanket_alerts` — verify chăn phải alert

### 8.2 Scan camera
```bash
python scripts/test_webcam.py --scan
```

### 8.3 Verify webcam FPS thực tế

Headless (không HDMI):
```bash
python scripts/test_webcam.py --source 0 --headless --save-snapshot events/webcam_test.jpg
```

Có HDMI/màn hình:
```bash
python scripts/test_webcam.py --source 0
```

Kỳ vọng FPS thực ≥ 21fps. Nếu thấp hơn:
- Cắm cổng USB 3.0 (cổng xanh, không phải đen)
- Cáp USB ngắn, chất lượng tốt
- Giảm resolution xuống 640x480

### 8.4 Benchmark pipeline

```bash
python scripts/benchmark.py --frames 100
```

Kỳ vọng trên RK3588 (CPU only, không NPU):
| Stage | Thời gian |
|---|---|
| MediaPipe (1280x720) | 25-40ms |
| Multi-signal detector (3 landmark mouth × 4 signal + 1 landmark nose × 4) | 6-12ms |
| YOLO CPU (chạy mỗi 5 frame, amortized) | 15-30ms |
| **End-to-end** | ~80-100ms (~10-12 FPS) |

> Note: pipeline hiện tại nặng hơn bản trước (~1 landmark + 1 signal) ~20ms vì
> dùng multi-signal voting + multi-landmark sampling cho ổn định. State machine
> + smoother chỉ cần ≥6 FPS để hoạt động đúng → 10-12 FPS dư an toàn.

Nếu chậm hơn:
- Giảm `CAMERA_WIDTH/HEIGHT` xuống `640`/`480` (giảm mediapipe ~50%)
- Tăng `YOLO_EVERY` lên `10`
- Tăng `CONFIRM_FRAMES` (ổn định hơn nhưng response chậm hơn)
- Setup NPU (mục 11) — tăng FPS YOLO ~10x

---

## 9. Chạy chương trình

### 9.1 Load `.env` và chạy

Tạo `/opt/baby-monitor/run.sh`:
```bash
#!/bin/bash
set -e
cd /opt/baby-monitor
set -a
source .env
set +a
source venv/bin/activate
exec python src/main.py
```

```bash
chmod +x /opt/baby-monitor/run.sh
./run.sh
```

### 9.2 Test 4 kịch bản cảnh báo

| # | Kịch bản | Kỳ vọng |
|---|---|---|
| 1 | **Mặt sạch ngồi yên 30s** | Debug bar votes = 0/4 hoặc 1/4. KHÔNG alert. |
| 2 | **Tay che mũi/miệng 15s+** | Votes tăng lên 2-3/4 (đặc biệt `lap` giảm rõ). Sau 15s → Telegram alert `MUI/MIENG BI CHE` |
| 3 | **Chăn phủ kín mặt 15s+** | Votes 4/4 ngay. Sau 15s → Telegram alert (hoặc `Mất hoàn toàn khuôn mặt` nếu mediapipe mất tracking) |
| 4 | **Rời khung (KHÔNG alert)** | Bước hẳn ra ngoài camera → YOLO không thấy người → state `NO_FACE`, KHÔNG gửi tin |

Mỗi alert lưu `events/possible_suffocation_risk_<timestamp>.jpg` + `.json` (kèm tất cả 4 signal values).

### 9.3 Recalibration (khi cần)

Calibration chỉ chạy 1 lần lúc khởi động. Nếu môi trường thay đổi (ánh sáng, vị trí trẻ, hoặc baseline drift sau lâu) → recalibrate:

- **GUI mode** (có HDMI): nhấn phím `R` trong cửa sổ video
- **Headless mode** (systemd, không HDMI): gửi tín hiệu SIGUSR1
  ```bash
  pgrep -f "src/main.py"   # tìm PID
  kill -USR1 <PID>          # trigger recalibrate
  ```
- **Auto-recalibrate**: app tự recal sau `AUTO_RECAL_AFTER_SEC` giây safe liên tục (default 30 phút). Tắt bằng `AUTO_RECAL_AFTER_SEC=0`.

Sau khi nhận signal, app log:
```
🔄 Yêu cầu recalibrate: SIGUSR1
🔄 Đã reset — sẽ calibrate lại khi thấy mặt.
✅ Calibration xong! q=0.92 | hist_thresh nose=0.84 mouth=0.81 | ...
```

### 9.4 Headless mode (không màn hình)

App đã có built-in HEADLESS support. Khi `HEADLESS=1` trong `.env`:
- Không gọi `cv2.imshow()` / `cv2.waitKey()` → không cần X server
- Hotkey `R` không hoạt động → dùng `kill -USR1 <pid>` để recalibrate (mục 9.3)

Production setup (systemd) BẮT BUỘC set `HEADLESS=1`. Nếu để `0` mà không có X → app crash với `cv2.error: cannot connect to X server`.

Nếu muốn xem live view khi debug, ssh X-forwarding:
```bash
ssh -X pi@<opi-ip>
HEADLESS=0 ./run.sh
```

---

## 10. Autostart bằng systemd

### 10.1 Tạo service file

```bash
sudo tee /etc/systemd/system/baby-monitor.service > /dev/null << 'EOF'
[Unit]
Description=Baby AI Alert Monitor
After=network.target

[Service]
Type=simple
User=pi
Group=video
WorkingDirectory=/opt/baby-monitor
EnvironmentFile=/opt/baby-monitor/.env
ExecStart=/opt/baby-monitor/venv/bin/python /opt/baby-monitor/src/main.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/baby-monitor.log
StandardError=append:/var/log/baby-monitor.log

[Install]
WantedBy=multi-user.target
EOF
```

### 10.2 Tạo file log + enable

```bash
sudo touch /var/log/baby-monitor.log
sudo chown pi:pi /var/log/baby-monitor.log

sudo systemctl daemon-reload
sudo systemctl enable baby-monitor
sudo systemctl start baby-monitor
sudo systemctl status baby-monitor
```

### 10.3 Theo dõi log live

```bash
journalctl -u baby-monitor -f
# hoặc
tail -f /var/log/baby-monitor.log
```

### 10.4 Log rotation (tránh đầy đĩa)

```bash
sudo tee /etc/logrotate.d/baby-monitor > /dev/null << 'EOF'
/var/log/baby-monitor.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0644 pi pi
    postrotate
        systemctl restart baby-monitor > /dev/null 2>&1 || true
    endscript
}
EOF
```

### 10.5 Dọn `events/` định kỳ

Cron job xóa snapshot cũ:
```bash
crontab -e
# Thêm dòng:
0 2 * * * find /opt/baby-monitor/events -mtime +30 -delete
```

---

## 11. (Optional) Tăng tốc bằng NPU RK3588

**Khi nào nên làm**: chỉ khi mục 8.4 benchmark cho end-to-end > 33ms (FPS < 30) và bạn cần real-time hơn.

RK3588 có **NPU 6 TOPS** dùng được cho YOLO inference, giảm thời gian từ ~80ms (CPU) xuống ~10ms (NPU).

### 11.1 Cài rknn-toolkit2 trên máy DEV (không phải OPi)

Convert YOLOv8n → RKNN format cần làm trên máy x86 Linux:
```bash
# Trên máy dev (Ubuntu x86_64)
pip install rknn-toolkit2 onnx
yolo export model=yolov8n.pt format=onnx
# Sau đó dùng rknn-toolkit2 convert .onnx → .rknn
# Theo doc: https://github.com/rockchip-linux/rknn-toolkit2
```

Output: `yolov8n.rknn` (~6MB).

### 11.2 Trên Orange Pi: cài rknn-lite runtime

```bash
pip install rknn-toolkit-lite2
```

### 11.3 Sửa code để dùng RKNN thay PyTorch

Cần viết wrapper `RKNNPersonDetector` trong [src/main.py](src/main.py) thay cho `PersonDetector` hiện tại. Logic giống nhau (`has_person()`), chỉ thay inference engine. Ping mình khi cần làm bước này.

---

## 12. (Optional) Kích relay khi có cảnh báo

Module relay 4 kênh bạn đang có (LED xanh) có thể đấu vào GPIO của OPi để:
- **Kênh 1**: bật còi báo động khi alert
- **Kênh 2**: bật đèn cảnh báo
- **Kênh 3**: kích thiết bị khác (quạt, monitor phụ huynh...)
- **Kênh 4**: dự phòng

### 12.1 Đấu dây

| Relay pin | OPi GPIO (BCM) | OPi physical pin |
|---|---|---|
| VCC | 5V | Pin 2 |
| GND | GND | Pin 6 |
| IN1 | GPIO 17 | Pin 11 |
| IN2 | GPIO 27 | Pin 13 |
| IN3 | GPIO 22 | Pin 15 |
| IN4 | GPIO 23 | Pin 16 |

> ⚠️ Relay 5V — OPi GPIO output 3.3V có thể không trigger được trên một số relay module. Nếu vậy → dùng relay 3.3V hoặc thêm transistor BJT/MOSFET đệm.

### 12.2 Cài thư viện GPIO

```bash
pip install OPi.GPIO
# Hoặc dùng gpiod (chuẩn kernel mới)
sudo apt install python3-libgpiod
```

### 12.3 Wire vào code

Trong `_dispatch_alert()` của [src/main.py](src/main.py), thêm trigger relay khi `should_alert`:

```python
import OPi.GPIO as GPIO   # hoặc gpiod
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

def _trigger_relay(channel=17, duration_sec=5):
    GPIO.output(channel, GPIO.HIGH)
    threading.Timer(duration_sec, lambda: GPIO.output(channel, GPIO.LOW)).start()
```

Gọi `_trigger_relay()` trong nhánh `if result.should_alert` của `run()`. Ping mình nếu cần làm hoàn chỉnh.

---

## 13. Troubleshooting

### `NumPy 2.x cannot be run...` / `opencv-python ... requires numpy>=2`
- numpy / opencv ABI mismatch. Mediapipe cần `numpy<2`, opencv ≥4.11 cần `numpy>=2` → xung đột.
- **Fix triệt để** (uninstall hết, cài lại đúng version):
  ```bash
  bash scripts/fix_env.sh
  ```
- Nếu script không có (deploy version cũ): chạy thủ công
  ```bash
  pip uninstall -y opencv-python opencv-contrib-python \
      opencv-python-headless opencv-contrib-python-headless
  pip install "numpy<2" "opencv-python<4.11" --force-reinstall
  ```

### App khởi động in `❌ OpenCV variant compile chống NumPy 2.x` rồi exit
- Guard tầng 2 fire — opencv variant đang cài chưa < 4.11
- Fix: `bash scripts/fix_env.sh`

### App khởi động in `❌ Phát hiện nhiều opencv variant cài cùng lúc`
- Có cả `opencv-python` và `opencv-contrib-python` cài → xung đột cv2 namespace
- Fix: `bash scripts/fix_env.sh` (sẽ uninstall hết rồi cài lại 1 cái)

### Tay che mũi/miệng NHƯNG không alert (false negative)
- Kiểm tra debug bar: dòng `MOUTH` khi tay che, giá trị `lap=` phải giảm xuống < 50 → vote
- Nếu `lap` của hand vẫn > 50: tăng `LAPVAR_ABSOLUTE_FLOOR` trong [src/occlusion_detector.py](src/occlusion_detector.py) (vd lên 80) → nhạy hơn
- Nếu `edge` của hand > 0.020: tăng `EDGE_ABSOLUTE_FLOOR` (vd lên 0.030)
- Nếu votes oscillate (lúc 2/4 lúc 1/4): giảm `CONFIRM_FRAMES` xuống `7-8`, tăng `SMOOTHER_MAX_MISS` lên `4-5`
- Recalibrate nếu baseline xấu (nhấn `R` hoặc `kill -USR1`)

### Mặt sạch nhưng vẫn alert (false positive)
- `quality` trong debug bar < 0.5 → calibration kém → recalibrate trong điều kiện ổn định
- Tăng `LAPVAR_ABSOLUTE_FLOOR` / `EDGE_ABSOLUTE_FLOOR` lên (bớt nhạy)
- Tăng `CONFIRM_FRAMES` lên `15-20` (cần che lâu hơn để alert)

### `Could not find a version that satisfies the requirement mediapipe`
- Python version sai (3.12+ chưa có wheel ARM64 cho mediapipe) → cài Python 3.11
- Pip outdated → `pip install --upgrade pip`

### `Killed` khi pip install (process bị OOM)
- Hết RAM → tạo swap (mục 3.4)
- Cài từng package một thay vì một lệnh dài

### `[ WARN:0 ] global cap_v4l.cpp ... can't open camera by index`
- User không thuộc group `video` → `sudo usermod -aG video $USER` + logout/login
- Camera bị app khác giữ → `sudo fuser -k /dev/video0`
- Sai index → `v4l2-ctl --list-devices`

### FPS thực < 15
- USB 2.0 thay vì 3.0 → đổi cổng (cổng xanh)
- Resolution quá cao → giảm xuống 640x480
- Thermal throttle (RK3588 nóng) → kiểm tra:
  ```bash
  cat /sys/class/thermal/thermal_zone0/temp
  # Nếu > 85000 (85°C) → cần tản nhiệt tốt hơn
  ```

### `cv2.error: ... cannot connect to X server`
- Đang chạy headless mà code có `cv2.imshow()` → set `HEADLESS=1` (xem mục 9.3)

### `Failed to send Telegram message`
- Mạng OPi không có internet → `ping 8.8.8.8`
- Token / chat ID sai → test bằng curl (mục 6.3)
- DNS lỗi → `sudo apt install -y resolvconf`

### systemd service không start
```bash
sudo systemctl status baby-monitor
journalctl -u baby-monitor -n 50
```
Thường vì:
- `.env` không đọc được (sai path / permission)
- venv path sai
- `User=pi` nhưng owner project khác → `sudo chown -R pi:pi /opt/baby-monitor`

### Sau vài giờ chạy ổn, đột nhiên crash
- SD card có bad block → check `dmesg | grep -i error`. Production nên dùng eMMC/NVMe
- Memory leak — set `MemoryMax=1G` trong systemd service để tự restart khi vượt
- Nhiệt độ — gắn quạt nếu chưa có

### MediaPipe load chậm (10-20s khi khởi động)
- Bình thường trên ARM lần đầu (download/extract model) → cache vào `~/.mediapipe/`
- Lần boot tiếp theo sẽ nhanh hơn (~3-5s)

---

## 14. Tổng kết stack

```
Orange Pi 5 (RK3588, 8GB RAM)
├── Ubuntu 22.04 Server ARM64
├── Logitech USB webcam (USB 3.0)
├── Module relay 4 kênh → GPIO (output cảnh báo cứng) — optional, mục 12
│
├── /opt/baby-monitor/
│   ├── venv/                     ← Python 3.11 isolated env
│   ├── requirements.txt          ← deps pinned (numpy<2, opencv<4.11)
│   ├── src/
│   │   ├── main.py               ← entrypoint, 3 guard env, signal handlers
│   │   ├── state_machine.py      ← FSM phát hiện ngạt thở
│   │   └── occlusion_detector.py ← Multi-signal voting (hist+skin+edge+lap_var)
│   ├── scripts/
│   │   ├── fix_env.sh            ← Uninstall opencv variants + cài đúng version
│   │   ├── test_webcam.py        ← Verify camera + FPS
│   │   └── benchmark.py          ← Đo pipeline
│   ├── tests/
│   │   ├── test_state_machine.py        (11 test)
│   │   └── test_occlusion_detector.py   (14 test)
│   ├── .env                      ← config (token, camera, threshold, HEADLESS=1)
│   ├── run.sh                    ← manual run wrapper
│   ├── events/                   ← snapshot + JSON khi alert (rotate 30 ngày)
│   ├── yolov8n.pt                ← person detection (hoặc .rknn nếu NPU)
│   └── INSTALL.md                ← file này
│
└── systemd services:
    ├── baby-monitor.service      ← autostart + restart on crash + SIGTERM clean
    ├── /etc/logrotate.d/...      ← rotate log 7 ngày
    └── cron: dọn events/         ← cleanup 2h sáng mỗi ngày
```

### Multi-signal detection (mục 8.1)

Mỗi frame, mỗi patch (mũi + 3 patch miệng) tính 4 signal:
- **Histogram HSV correlation** — color distribution similarity với baseline
- **Skin pixel ratio** — % pixel match skin HSV range (chăn/gối có skin% thấp)
- **Edge density** (Canny) — % pixel có edge (chăn phẳng → ít edge)
- **Laplacian variance** — variance của Laplacian filter (mặt có chi tiết nhỏ → high; tay back phẳng → low)

Vote rule: ≥ 2/4 signal trong patch → occluded. **Bất kỳ** patch nào (mũi hoặc miệng) occluded → fire.

Threshold ADAPTIVE từ calibration (mean - 3σ cho hist) + ABSOLUTE FLOOR cho edge (0.020) và lap_var (50) → tay (texture thấp) luôn vote dù baseline yếu.

---

## 15. Báo lỗi

Khi gặp vấn đề:
```bash
# Thu thập thông tin
cat /proc/device-tree/model > diag.txt
free -h >> diag.txt
v4l2-ctl --list-devices >> diag.txt
python --version >> diag.txt
pip list >> diag.txt
journalctl -u baby-monitor -n 100 >> diag.txt
python scripts/benchmark.py --frames 50 >> diag.txt 2>&1
```

Gửi `diag.txt` + mô tả tình huống.
