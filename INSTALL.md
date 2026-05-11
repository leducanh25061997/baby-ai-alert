# Hướng dẫn cài đặt Baby AI Alert

Hệ thống phát hiện nguy cơ ngạt thở ở trẻ qua camera, gửi cảnh báo Telegram.

## Mục lục

1. [Yêu cầu phần cứng](#1-yêu-cầu-phần-cứng)
2. [Cài driver NVIDIA + CUDA](#2-cài-driver-nvidia--cuda) (chỉ khi có GPU)
3. [Cài Python + dependencies](#3-cài-python--dependencies)
4. [Tạo Telegram bot + lấy token](#4-tạo-telegram-bot--lấy-token)
5. [Cấu hình bằng biến môi trường](#5-cấu-hình-bằng-biến-môi-trường)
6. [Verify cài đặt](#6-verify-cài-đặt)
7. [Chạy chương trình](#7-chạy-chương-trình)
8. [Đóng gói production (autostart)](#8-đóng-gói-production-autostart)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Yêu cầu phần cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| **CPU** | x86 4 core 2.0GHz | x86 8 core 3.0GHz+ |
| **RAM** | 4 GB | 8 GB+ |
| **GPU** | Không bắt buộc | NVIDIA Quadro/GTX/RTX, ≥2GB VRAM |
| **Camera** | USB webcam 720p | Logitech C270/C920 / IP camera RTSP có IR night-vision |
| **OS** | Windows 10/11, Ubuntu 20.04+ | Windows 11 |
| **Python** | 3.10 | 3.10 - 3.12 (3.14 chưa được mediapipe support đầy đủ) |
| **Mạng** | Ổn định để gửi Telegram | |

> **Lưu ý quan trọng về Python**: MediaPipe hiện tại chỉ có wheel sẵn cho **Python 3.10 - 3.12**. Nếu máy bạn đang dùng Python 3.13+ → cài thêm bản 3.12 song song.

---

## 2. Cài driver NVIDIA + CUDA

> Bỏ qua mục này nếu chỉ chạy CPU. Pipeline vẫn hoạt động không có GPU, chỉ chậm hơn.

### 2.1 Cài driver

1. Vào https://www.nvidia.com/Download/index.aspx
2. Chọn đúng dòng card (vd: **Quadro P400 / P620 / T400**)
3. Tải Studio Driver hoặc Game Ready Driver, cài → **reboot**
4. Verify:
   ```powershell
   nvidia-smi
   ```
   Phải thấy: tên card, CUDA version (góc phải trên), driver version.

### 2.2 Không cần cài CUDA Toolkit riêng

PyTorch wheel đã đóng gói sẵn CUDA runtime. Chỉ cần driver.

---

## 3. Cài Python + dependencies

### 3.1 Cài Python 3.12 (khuyến nghị)

Tải https://www.python.org/downloads/release/python-3128/ → chọn **"Add to PATH"** lúc cài.

Verify:
```powershell
py -3.12 --version
```

### 3.2 Tạo virtual environment

```powershell
cd d:\AI-AGENT\baby-ai-alert
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

Sau khi `activate`, prompt sẽ có `(venv)` ở đầu.

### 3.3 Cài deps cơ bản

```powershell
pip install -r requirements.txt
```

File này cài: `opencv-python`, `mediapipe`, `numpy`, `python-telegram-bot`, `ultralytics`.

### 3.4 Cài PyTorch CUDA (nếu có GPU)

`ultralytics` mặc định kéo PyTorch CPU. Để dùng GPU cần thay bằng bản CUDA:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA hoạt động:
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Phải in `CUDA available: True` và tên card.

### 3.5 Set encoding UTF-8 cho terminal

Windows console mặc định không hiển thị emoji → log của chương trình sẽ crash. Set 1 lần cho mỗi session:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

Hoặc set vĩnh viễn trong PowerShell profile (`$PROFILE`).

---

## 4. Tạo Telegram bot + lấy token

### 4.1 Tạo bot

1. Mở Telegram, tìm `@BotFather`
2. Gõ `/newbot`, đặt tên, đặt username (kết thúc bằng `bot`)
3. BotFather trả về **token** dạng `1234567890:ABC...` → copy lại

### 4.2 Lấy chat ID

1. Gõ vài tin nhắn cho bot vừa tạo
2. Truy cập (thay TOKEN):
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
3. Trong JSON trả về tìm `"chat":{"id":7316578932,...}` → đó là **chat ID**

### 4.3 Test gửi tin

```powershell
$token = "TOKEN_CỦA_BẠN"
$chat  = "CHAT_ID"
curl "https://api.telegram.org/bot$token/sendMessage?chat_id=$chat&text=test"
```

Bot phải gửi "test" về Telegram của bạn.

---

## 5. Cấu hình bằng biến môi trường

Tất cả config đều có default — chỉ override khi cần.

| Biến | Default | Mô tả |
|---|---|---|
| `TELEGRAM_TOKEN` | hardcode trong code | Token bot Telegram |
| `TELEGRAM_CHAT_ID` | hardcode | Chat ID nhận cảnh báo |
| `CAMERA_SOURCE` | `0` | USB index (`0`, `1`...) hoặc `rtsp://user:pass@ip:554/stream` |
| `CAMERA_WIDTH` | `1280` | Resolution rộng |
| `CAMERA_HEIGHT` | `720` | Resolution cao |
| `CAMERA_FPS` | `30` | Target FPS |
| `YOLO_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `YOLO_EVERY` | `0` (auto) | Chạy YOLO mỗi N frame; `0` = auto (2 nếu CUDA, 5 nếu CPU) |

### Set tạm thời (cho session hiện tại)

```powershell
$env:TELEGRAM_TOKEN = "1234567890:ABC..."
$env:TELEGRAM_CHAT_ID = "7316578932"
$env:CAMERA_SOURCE = "0"
$env:YOLO_DEVICE = "auto"
$env:PYTHONIOENCODING = "utf-8"
```

### Set vĩnh viễn — dùng file `.env`

Tạo `D:\AI-AGENT\baby-ai-alert\.env`:
```
TELEGRAM_TOKEN=1234567890:ABC...
TELEGRAM_CHAT_ID=7316578932
CAMERA_SOURCE=0
YOLO_DEVICE=auto
PYTHONIOENCODING=utf-8
```

Tạo `run.ps1`:
```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2]
    }
}
& venv\Scripts\python.exe src\main.py
```

Sau này chỉ cần `.\run.ps1`.

> ⚠️ **Đừng commit `.env` lên git** — thêm `.env` vào `.gitignore`.

---

## 6. Verify cài đặt

Chạy theo thứ tự, mỗi bước phải pass mới qua bước sau.

### 6.1 Test state machine (không cần camera/GPU/network)

```powershell
python tests\test_state_machine.py
```

Kỳ vọng: `🎉 11/11 test PASS`. Nếu fail → báo cho dev (state machine logic sai).

### 6.2 Scan camera

```powershell
python scripts\test_webcam.py --scan
```

Liệt kê index camera khả dụng. Webcam Logitech thường ở index 0 (hoặc 1 nếu có camera laptop).

### 6.3 Verify webcam chạy đúng resolution + FPS

```powershell
python scripts\test_webcam.py --source 0
```

Mở cửa sổ live 8s, đo FPS thực. Kỳ vọng FPS thực ≥ 21fps (70% của 30).

Nếu headless server (không màn hình):
```powershell
python scripts\test_webcam.py --source 0 --headless --save-snapshot events\webcam_test.jpg
```

### 6.4 Benchmark pipeline

```powershell
# Đo full pipeline
python scripts\benchmark.py --frames 200

# So sánh CPU vs GPU
python scripts\benchmark.py --frames 100 --yolo-device cpu
python scripts\benchmark.py --frames 100 --yolo-device cuda
```

Kỳ vọng (Quadro P400 + Logitech C270):
| Stage | CPU | CUDA |
|---|---|---|
| MediaPipe | 8-15ms | (CPU only) |
| Histogram | 1-3ms | - |
| YOLO | 80-120ms | 8-15ms |
| **End-to-end** | ~110ms (~9 FPS) | ~30ms (~30 FPS) |

Nếu end-to-end > 33ms → script tự suggest fix (giảm res, throttle YOLO, dùng TensorRT...).

---

## 7. Chạy chương trình

```powershell
venv\Scripts\activate
python src\main.py
```

### Quy trình:
1. Chương trình mở camera, log resolution + FPS thực tế
2. Chờ phát hiện mặt → in `✅ Phát hiện mặt!`
3. Calibrate 5s — **giữ nguyên mặt trẻ**, không di chuyển
4. In `✅ Calibration xong! Bắt đầu giám sát.`
5. Live: hiển thị landmarks mũi/miệng + correlation
6. Khi che mũi/miệng → countdown 15s → gửi Telegram + lưu vào `events/`

### Test cảnh báo:
1. Lấy khăn/tay che mũi-miệng (vẫn để mặt trong khung) → đợi 15s → phải nhận tin Telegram (`MUI/MIENG BI CHE`)
2. Lấy chăn phủ kín cả mặt → đợi 15s → tin Telegram (`Mất hoàn toàn khuôn mặt`)
3. Bước hẳn ra ngoài camera → state về `NO_FACE`, **không** gửi tin

Mỗi lần alert lưu 1 file `events/possible_suffocation_risk_<timestamp>.jpg` + `.json`.

### Thoát: nhấn `Q` trong cửa sổ camera.

---

## 8. Đóng gói production (autostart)

### Windows — dùng Task Scheduler

1. Tạo file `D:\AI-AGENT\baby-ai-alert\start.bat`:
   ```bat
   @echo off
   cd /d D:\AI-AGENT\baby-ai-alert
   call venv\Scripts\activate
   set PYTHONIOENCODING=utf-8
   python src\main.py >> logs\out.log 2>&1
   ```
2. Tạo folder `logs\`
3. Mở **Task Scheduler** → **Create Basic Task**:
   - Trigger: **At log on** (hoặc **At startup**)
   - Action: Start a program → `D:\AI-AGENT\baby-ai-alert\start.bat`
   - Settings: ✅ "Restart task if fails" (sau 1 phút, 3 lần)

### Linux — systemd service

```ini
# /etc/systemd/system/baby-monitor.service
[Unit]
Description=Baby AI Alert Monitor
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/opt/baby-monitor
EnvironmentFile=/opt/baby-monitor/.env
ExecStart=/opt/baby-monitor/venv/bin/python /opt/baby-monitor/src/main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable baby-monitor
sudo systemctl start baby-monitor
sudo systemctl status baby-monitor
journalctl -u baby-monitor -f   # xem log live
```

### Lưu ý production:
- **Log rotation**: file log dài hạn sẽ đầy disk → setup `logrotate` (Linux) hoặc xóa thủ công (Windows)
- **Cleanup events/**: tạo script cron xóa file > 30 ngày
- **NTP**: đồng bộ thời gian (`w32tm` Windows / `chronyd` Linux) để timestamp chính xác

---

## 9. Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'mediapipe'`
- Chưa activate venv → chạy `venv\Scripts\activate`
- Hoặc Python version > 3.12 → cài lại với 3.12

### Lỗi: `ImportError: DLL load failed` khi import cv2 (Windows)
- Cài Microsoft Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe

### `nvidia-smi` không tìm thấy
- Driver chưa cài hoặc cài lỗi → cài lại + reboot
- Verify card được nhận: Device Manager → Display adapters

### `torch.cuda.is_available()` trả về `False`
- Kiểm tra `nvidia-smi` chạy được — nếu không thì lỗi driver
- Đang cài bản torch CPU → cài lại với `--index-url https://download.pytorch.org/whl/cu121`
- Card quá cũ (Compute Capability < 5.0) → cần torch+cu118 thay vì cu121

### Camera không mở được — `❌ Không mở được camera: 0`
- Camera đang bị app khác giữ (Skype, Zoom, Teams...) → tắt
- Sai index → `python scripts\test_webcam.py --scan`
- Trên Linux: thiếu quyền → `sudo usermod -a -G video $USER` và logout/login

### FPS thực tế thấp hơn target nhiều
- USB 2.0 không đủ băng thông cho 720p MJPG → cắm cổng USB 3.0 (cổng xanh)
- Cáp USB dài/kém → đổi cáp ngắn hơn
- Camera không hỗ trợ resolution → giảm:
  ```powershell
  $env:CAMERA_WIDTH  = "640"
  $env:CAMERA_HEIGHT = "480"
  ```

### Telegram không gửi được — `❌ Lỗi Telegram: ...`
- Token sai → kiểm tra `getUpdates` URL
- Chat ID sai → bot phải nhận được ít nhất 1 tin trước khi `getUpdates` có chat
- Tường lửa chặn — test bằng `curl https://api.telegram.org`

### Calibration không kết thúc — kẹt ở `CALIBRATING`
- Mặt không cố định trong 5s → giữ yên
- Ánh sáng quá tối → mediapipe không detect được landmark
- Kiểm tra `min_detection_confidence` trong [src/main.py](src/main.py) → giảm xuống 0.3 nếu cần

### Báo nhầm liên tục (false alert) khi trẻ chỉ cử động đầu
- Threshold quá cao → giảm `HIST_CORR_THRESHOLD` từ 0.65 xuống 0.50 trong [src/main.py](src/main.py)
- Hoặc tăng `CONFIRM_FRAMES` từ 15 lên 30 (cần lâu hơn để confirm)

### Bỏ sót khi chăn phủ một phần
- Đảm bảo dùng OR-logic ở [src/main.py:131-133](src/main.py#L131-L133) (`nose_occ or mouth_occ`)
- Test bằng `python tests\test_state_machine.py` để verify state machine vẫn đúng

### YOLO báo `RuntimeError: CUDA out of memory`
- Card 2GB VRAM bị tràn — đổi sang `YOLO_DEVICE=cpu`, hoặc dùng model nhỏ hơn

---

## 10. Cấu trúc project

```
baby-ai-alert/
├── INSTALL.md              ← file này
├── requirements.txt
├── yolov8n.pt              ← YOLOv8 nano cho person detection
├── face_mask.pt            ← (chưa dùng, dự phòng)
├── src/
│   ├── main.py             ← chương trình chính
│   └── state_machine.py    ← logic phát hiện ngạt thở (testable)
├── scripts/
│   ├── test_webcam.py      ← verify webcam
│   └── benchmark.py        ← đo FPS pipeline
├── tests/
│   └── test_state_machine.py
├── events/                 ← snapshot + JSON khi có cảnh báo
└── venv/                   ← Python virtual env (gitignore)
```

---

## 11. Liên hệ / báo lỗi

Khi gặp vấn đề chưa có trong troubleshooting:
1. Chạy `python scripts\benchmark.py --frames 50` và copy output
2. Chụp ảnh log của `python src\main.py`
3. `nvidia-smi` (nếu có GPU)
4. Gửi cho dev cùng mô tả tình huống cụ thể.
