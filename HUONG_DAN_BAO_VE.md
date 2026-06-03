# Hướng dẫn bảo vệ đồ án — Baby AI Alert

> **Cách dùng tài liệu này**:
> - Phần 1-4: học để **hiểu sâu**, nói được bằng lời của mình.
> - Phần 5: bộ câu hỏi + câu trả lời mẫu, đọc đi đọc lại để **phản xạ**.
> - Phần 6: kịch bản demo, luyện trước **3-5 lần** ở nhà.
> - Phần 7-8: bài thuộc khi hội đồng hỏi câu **khó / câu bẫy**.

---

## Mục lục

1. [Tóm tắt đồ án trong 30 giây](#1-tóm-tắt-đồ-án-trong-30-giây)
2. [Cấu trúc project — sơ đồ nhà](#2-cấu-trúc-project--sơ-đồ-nhà)
3. [Luồng xử lý 1 khung hình (Pipeline)](#3-luồng-xử-lý-1-khung-hình-pipeline)
4. [Giải thích thuật toán bằng ví dụ trực quan](#4-giải-thích-thuật-toán-bằng-ví-dụ-trực-quan)
5. [Bộ câu hỏi — câu trả lời chuẩn (50 câu)](#5-bộ-câu-hỏi--câu-trả-lời-chuẩn-50-câu)
6. [Kịch bản demo trước hội đồng (5 phút)](#6-kịch-bản-demo-trước-hội-đồng-5-phút)
7. [Các điểm nhấn để gây ấn tượng](#7-các-điểm-nhấn-để-gây-ấn-tượng)
8. [Câu khó / câu bẫy — cách trả lời an toàn](#8-câu-khó--câu-bẫy--cách-trả-lời-an-toàn)

---

## 1. Tóm tắt đồ án trong 30 giây

> *Câu mở đầu khi giới thiệu với hội đồng. Học thuộc, nói trơn tru.*

**"Đề tài của em là hệ thống cảnh báo che mũi/miệng ở trẻ sơ sinh bằng AI, chạy trên máy tính nhúng Raspberry Pi 4. Hệ thống dùng camera quan sát trẻ liên tục, dùng MediaPipe để xác định vị trí mũi và miệng, sau đó dùng 4 tín hiệu hình ảnh để bỏ phiếu xem mũi/miệng có bị vật lạ che hay không. Nếu bị che quá 15 giây thì gửi cảnh báo về Telegram cho cha mẹ kèm ảnh chụp tại thời điểm đó. Toàn bộ chạy CPU-only trên Raspberry Pi 4, không cần GPU, hoạt động real-time khoảng 6-8 khung hình mỗi giây ở độ phân giải 640×480."**

### 4 điểm mạnh để khoe ngay:
1. **Real-time trên thiết bị nhúng giá rẻ** (Raspberry Pi 4 ~1.5 triệu VND) — không cần cloud, không cần GPU
2. **Multi-signal voting** — 4 tín hiệu bỏ phiếu chéo, không phụ thuộc 1 model AI duy nhất → robust
3. **Có state machine + grace period** — tránh false alert khi mặt bị che trong tích tắc
4. **Thiết kế safety-first + YOLO hỗ trợ** — khi mất mặt thì ưu tiên cảnh báo (thà báo nhầm còn hơn bỏ lọt ca bị che); YOLO giúp bắt đầu đếm khi thấy người và im lặng khi khung trống

---

## 2. Cấu trúc project — sơ đồ nhà

Hình dung project như **một ngôi nhà 3 phòng**:

```
baby-ai-alert/
│
├── src/                              ← "Khu vực chính" — code chạy app
│   ├── main.py                       ← Phòng khách: tiếp nhận camera, điều phối
│   ├── state_machine.py              ← Phòng não: quyết định khi nào alert
│   ├── occlusion_detector.py         ← Phòng mắt: nhìn và phân tích pixel mũi/miệng
│   ├── scene_monitor.py              ← Phòng cảm biến: blur-gate + motion + luma
│   └── alert_policy.py               ← Phòng đồng hồ: timing watchdog/heartbeat/nhắc hiệu chỉnh
│
├── tests/                            ← Kho kiểm thử (59 unit test)
│   ├── test_state_machine.py         (14 test cho FSM)
│   ├── test_occlusion_detector.py    (16 test cho detector)
│   ├── test_scene_monitor.py         (7 test cho blur/motion/luma)
│   └── test_alert_policy.py          (14 test cho timing policy)
│
├── scripts/                          ← Tủ công cụ
│   ├── install_pi4.sh                ← Tự động cài trên Raspberry Pi 4
│   ├── fix_env.sh / fix_env.ps1      ← Sửa môi trường Python bị lỗi
│   ├── test_webcam.py                ← Test camera
│   └── benchmark.py                  ← Đo FPS pipeline
│
├── events/                           ← Nhật ký cảnh báo (ảnh + JSON)
│
├── yolov8n.pt                        ← Model AI nhận diện người (6MB)
├── requirements.txt                  ← Danh sách thư viện cần cài
└── INSTALL_PI4.md                    ← Hướng dẫn cài trên Raspberry Pi 4
```

### 3 file source quan trọng nhất — em phải hiểu rõ:

| File | Vai trò | Số dòng |
|---|---|---|
| **main.py** | Entry point: mở camera, gọi MediaPipe, ghép tất cả lại, gửi Telegram | ~700 |
| **state_machine.py** | Não bộ ra quyết định: SAFE / ALERT / NO_FACE / CALIBRATING | ~140 |
| **occlusion_detector.py** | Mắt: phân tích pixel ở mũi/miệng để bỏ phiếu "có bị che không" | ~420 |

### Tại sao chia 3 file mà không gộp 1?

**Câu trả lời chuẩn**: *"Em chia theo nguyên tắc Separation of Concerns. State machine không cần biết về OpenCV hay MediaPipe, nó chỉ nhận input boolean và quyết định state. Detector không cần biết về Telegram hay camera. Tách ra giúp em viết unit test cho state machine mà không cần camera thật — em có 59 test pass 100%. Nếu mai sau đổi từ MediaPipe sang model khác, em chỉ cần thay detector, state machine không phải đụng."*

---

## 3. Luồng xử lý 1 khung hình (Pipeline)

> *Đây là phần BẮT BUỘC nói rõ ràng. Hội đồng sẽ hỏi: "Em mô tả cho thầy/cô luồng xử lý 1 khung hình từ lúc camera bắt được đến lúc cảnh báo Telegram."*

### Sơ đồ pipeline (vẽ ra giấy/slide khi bảo vệ):

```
┌─────────────────┐
│  Camera USB     │  15 FPS, 640x480
│  (Logitech)     │
└────────┬────────┘
         ▼
  ┌─────────────┐
  │ cv2.read()  │  Lấy 1 khung BGR
  └──────┬──────┘
         ▼
  ┌──────────────────────────┐
  │ MediaPipe FaceMesh       │  ~60-90ms
  │ → tìm 468 landmark trên  │
  │   khuôn mặt              │
  └────────┬─────────────────┘
           │
    ┌──────┴──────┐
    │             │
  CÓ MẶT       MẤT MẶT
    │             │
    ▼             ▼
┌─────────────┐ ┌────────────────┐
│ Detector    │ │ YOLO check     │
│ 4-signal    │ │ "có người      │
│ voting      │ │  trong khung?" │
│             │ │  ~40-80ms(CPU) │
│ ~6-12ms     │ └────────┬───────┘
└──────┬──────┘          │
       │                 │
       └────────┬────────┘
                ▼
       ┌─────────────────┐
       │ State Machine   │  ← QUYẾT ĐỊNH ALERT/SAFE
       │  + Smoother     │     dựa trên 15 giây liên tục
       └────────┬────────┘
                ▼
        ┌──────────────┐
        │  ALERT?      │
        └──┬───────┬───┘
           │       │
        Không    Có
           │       │
           │       ▼
           │  ┌──────────────────────┐
           │  │ 1. Lưu ảnh vào events│
           │  │ 2. Lưu JSON metadata │
           │  │ 3. Gửi Telegram      │
           │  └──────────────────────┘
           ▼
       Tiếp tục
       frame sau
```

### Diễn giải từng bước (cho slide):

1. **Đọc camera** (~3ms): OpenCV đọc 1 khung hình 640×480 từ webcam USB.
2. **MediaPipe FaceMesh** (~60-90ms trên Pi 4): Google MediaPipe phát hiện 468 điểm landmark trên khuôn mặt (mắt, mũi, miệng, cằm...). Em chỉ dùng 4 điểm: 1 ở mũi (`NOSE_TIP=4`), 3 ở miệng (lip trên 13, giữa 14, lip dưới 17).
3. **Cắt patch quanh landmark** (~1ms): mỗi điểm landmark, em cắt 1 ô vuông 70×70 pixel xung quanh. Đó là khu vực mũi/miệng cần giám sát.
4. **Tính 4 tín hiệu cho mỗi patch** (~8ms):
   - Histogram HSV (phân bố màu)
   - Tỷ lệ pixel da (skin ratio)
   - Mật độ cạnh (edge density bằng Canny)
   - Phương sai Laplacian (lap variance = độ chi tiết)
5. **Bỏ phiếu**: mỗi tín hiệu nói "có/không bị che". Nếu ≥2/4 phiếu nói "có" → patch đó bị che.
6. **Smoother** (lọc nhiễu): cần CÓ ≥7/10 frame liên tiếp nói "bị che" mới chính thức coi là bị che → tránh false alert do landmark nhảy 1 frame.
7. **State machine**: nếu đã ở trạng thái bị che trong 15 giây liên tục → **ALERT**.
8. **Gửi cảnh báo**:
   - Lưu ảnh JPG + JSON metadata vào `events/`
   - Gửi ảnh + caption qua Telegram Bot API (chạy ở thread riêng để không chặn pipeline)

### Em chỉ cần thuộc 5 con số:

| Thông số | Giá trị | Ý nghĩa |
|---|---|---|
| **FPS pipeline** | 6-8 FPS | Tốc độ thực tế trên Raspberry Pi 4 (640×480) |
| **Ngưỡng alert** | 15 giây | Bị che liên tục bao lâu thì cảnh báo |
| **Calibration** | 5 giây | Thời gian học baseline khuôn mặt trẻ |
| **Smoother** | 7/10 frame | Lọc nhiễu trước state machine |
| **Cooldown** | 60 giây | Không gửi alert trùng trong 60s |

---

## 4. Giải thích thuật toán bằng ví dụ trực quan

### 4.1 MediaPipe FaceMesh — "định vị mũi và miệng"

**Cách giải thích cho hội đồng**:

> *"MediaPipe FaceMesh là model học sâu của Google, được train trên hàng triệu khuôn mặt. Em đưa 1 ảnh vào, nó trả về 468 điểm 3D đánh dấu cấu trúc khuôn mặt. Em chỉ quan tâm 4 điểm: đầu mũi và 3 điểm trên miệng. Em chọn MediaPipe vì nó đã được tối ưu cho thiết bị di động và nhúng — chạy ~60-90ms/frame trên CPU Raspberry Pi 4 ở 640×480, đủ nhanh cho real-time."*

**Tại sao không tự train model?**
> *"Em không có dataset đủ lớn và thời gian training. MediaPipe là solution mature, free, chính xác >95% trên benchmark phổ thông. Đề tài tập trung vào KỸ THUẬT PHÁT HIỆN BỊ CHE chứ không phải training face detector."*

### 4.2 Tại sao cần 4 tín hiệu, không phải 1?

**Đây là điểm KHOE NHẤT của đồ án. Học thuộc luôn:**

> *"Mỗi tín hiệu có 1 điểm yếu riêng, không cái nào hoàn hảo. Em dùng 4 tín hiệu vote chéo để bù trừ lẫn nhau:"*

| Tín hiệu | Phát hiện tốt | Bị bịt mắt khi |
|---|---|---|
| **Histogram (màu sắc)** | Chăn xanh, gối đỏ phủ mặt → màu thay đổi | Tay che mặt — tay cùng tone da với mặt |
| **Skin ratio (tỷ lệ da)** | Chăn/gối không có pixel da | Tay che mặt — tay LÀ da |
| **Edge density (cạnh)** | Chăn phẳng → ít đường viền | Tay có nếp gấp, móng tay → vẫn nhiều cạnh |
| **Laplacian variance (độ chi tiết)** | Lưng bàn tay phẳng, ít chi tiết | (yếu nhất khi ánh sáng đổi đột ngột) |

**Câu phản hồi nếu hội đồng hỏi "tại sao thiết kế thế này"**:
> *"Em đã test thực tế. Nếu chỉ dùng histogram (cách truyền thống), tay che mặt sẽ không phát hiện được — đây là failure mode nguy hiểm vì che mũi/miệng do trẻ tự đè tay lên mặt là kịch bản thực tế. Em phát hiện Laplacian variance là discriminator MẠNH NHẤT cho tình huống này: mặt có môi, lỗ mũi, lông mày → lap_var khoảng 1500. Lưng bàn tay phẳng → lap_var chỉ 3-5. Tỷ lệ chênh tới ~400 lần. Đây là phát hiện riêng em rút ra khi quan sát data."*

### 4.3 Vote rule — bỏ phiếu chéo

```
Patch mũi: 4 phiếu (hist/skin/edge/lap)
Patch miệng: 4 phiếu

Quyết định: BẤT KỲ patch nào (mũi HOẶC miệng) có ≥2/4 phiếu → coi là bị che
```

**Tại sao 2/4 mà không phải 3/4 hay 4/4?**
> *"Em chọn 2/4 sau khi cân nhắc trade-off: 1/4 quá lỏng → nhiều false alert. 3/4 quá chặt → bỏ sót case tay che (chỉ có lap_var và edge vote, skin và hist không vote vì tay là da). 2/4 là điểm cân bằng cho phép phát hiện cả chăn (4/4 vote) và tay (vừa đủ 2/4 nhờ lap_var + edge)."*

**Tại sao BẤT KỲ patch nào, không phải CẢ HAI?**
> *"Vì trong tình huống thực tế, đôi khi miệng bị che mà mũi vẫn lộ một phần, hoặc ngược lại. Quan trọng là EM CẢNH BÁO KỊP THỜI — an toàn của trẻ. Em chọn safety-first: 1 trong 2 patch nghi ngờ → alert."*

### 4.4 Calibration — "học baseline khuôn mặt"

**Vấn đề**: Mỗi đứa trẻ có tone da khác nhau, ánh sáng phòng khác nhau. Threshold cố định không hoạt động.

**Giải pháp**: Trong 5 giây đầu, app GHI NHỚ baseline mặt sạch:
- Trung bình histogram (mean color distribution)
- Trung bình skin ratio
- Trung bình edge density
- Trung bình Laplacian variance

Sau đó tính threshold **ADAPTIVE** dựa trên phân bố quan sát:
- `hist_threshold = mean_corr - 3 × stddev` (~99% confidence interval)
- `skin_min = mean_skin × 0.55` (báo khi giảm 45%)
- `edge_min = mean_edge × 0.70` (báo khi giảm 30%)
- `lap_var_min = mean_lap × 0.50` (báo khi giảm 50%)

**Câu mẫu giải thích**:
> *"Em không hardcode threshold mà tính từ data thực tế của từng phiên. Mỗi user có baseline riêng. Hệ thống tự tính ngưỡng dựa trên độ ổn định (standard deviation) của các sample calibration."*

### 4.5 Smoothing buffer — "lọc nhiễu"

**Vấn đề**: MediaPipe đôi khi nhảy landmark 1-2 frame → false positive.

**Giải pháp**: Cần ≥7 frame trong 10 frame liên tiếp đồng ý "bị che" mới chính thức coi là bị che. Cho phép tới 3 frame nhiễu.

```python
class SmoothingBuffer:
    confirm = 10        # cửa sổ 10 frame
    max_miss = 3        # cho phép 3 frame sai
    # Cần ≥7/10 vote True mới confirm
```

### 4.6 State Machine — "não bộ ra quyết định"

**4 trạng thái**:

| Trạng thái | Khi nào | Hành động |
|---|---|---|
| **CALIBRATING** | 5 giây đầu khi thấy mặt | Học baseline |
| **NO_FACE** | Không thấy mặt + YOLO cũng không thấy người | Chờ (trẻ ra ngoài khung) |
| **SAFE** | Thấy mặt, vote < 2/4 | Giám sát bình thường |
| **ALERT** | Vote ≥ 2/4 đã ≥ 15 giây | Gửi Telegram + lưu event |

**3 trigger dẫn đến ALERT**:
1. `TRIGGER_HISTOGRAM` — vote 4 tín hiệu phát hiện bị che (chăn, gối, tay...)
2. `TRIGGER_FACE_LOST` — MẤT HOÀN TOÀN mặt nhưng YOLO khẳng định vẫn có người trong khung → suy ra trẻ bị phủ kín
3. (Bonus) Hết grace 1.5 giây sau khi mặt biến mất → bắt đầu đếm về `face_lost`

**Câu thoại giải thích**:
> *"Em không gửi alert ngay khi phát hiện che — có thể là che chớp 1 giây không sao. Em đếm 15 giây liên tục. Trong 15 giây đó, nếu mặt LẠI XUẤT HIỆN và sạch trở lại, đếm reset về 0. Đây là cơ chế tránh false alert quan trọng nhất."*

### 4.7 YOLO — vai trò phụ nhưng cốt lõi

**Tại sao có YOLO?**

> *"YOLO không phát hiện bị che — đó là việc của detector chính. YOLO là bộ check PHỤ cho tình huống KHÔNG thấy mặt. Nhưng em theo nguyên tắc SAFETY-FIRST: khi đang thấy mặt mà mặt biến mất → mặc định coi là NGHI BỊ PHỦ KÍN và đếm để báo, KHÔNG để YOLO=False hủy cảnh báo. Vì sao? Camera top-down, trẻ nằm bị chăn phủ → YOLO (train trên người đứng) hay sót → nếu tin 'YOLO không thấy = rời khung' thì BỎ LỌT ca bị che thật.*
> *YOLO chỉ giúp 2 việc an toàn:*
> - *YOLO thấy CÓ người mà không thấy mặt (trẻ chưa từng track mặt) → BẮT ĐẦU đếm.*
> - *Khung trống, CHƯA TỪNG thấy mặt → im lặng (NO_FACE), không báo nhầm khi chưa đặt trẻ vào."*
>
> *"Đánh đổi em chấp nhận: nếu cha mẹ bế trẻ ra khỏi khung >15s thì vẫn có 1 cảnh báo nhầm — em ưu tiên KHÔNG BAO GIỜ MISS ca bị che hơn là tránh vài false alarm."*

**Tại sao YOLOv8n (nano) mà không phải s/m/l?**
> *"Nano là phiên bản nhỏ nhất, 6MB, nhẹ nhất cho CPU. s/m/l chậm hơn nhiều, không real-time được trên Raspberry Pi 4 (không có NPU). Em không cần độ chính xác cao, chỉ cần biết 'có thân người' hay không."*

**Tại sao không chạy YOLO mỗi frame?**
> *"Để tối ưu CPU. Trên Pi 4 em set `YOLO_EVERY=10` — chỉ chạy 1 lần mỗi 10 frame, cache kết quả. Lý do: phân biệt 'rời khung' vs 'bị phủ' không cần real-time — 1-2 lần/giây là dư."*

---

## 5. Bộ câu hỏi — câu trả lời chuẩn (50 câu)

> *Học thuộc các câu in **đậm**. Câu khó có gạch dưới — luyện trả lời ngắn gọn, đừng vòng vo.*

### A. Câu hỏi về mục đích đề tài (5 câu)

**Q1. Em làm đề tài này giải quyết bài toán gì?**

A: *"Em giải quyết bài toán phát hiện sớm khi mũi/miệng trẻ sơ sinh bị che (bởi chăn/gối/tay/vật lạ) — một yếu tố nguy cơ gây ngạt khi ngủ. Theo WHO, SIDS (Hội chứng đột tử ở trẻ sơ sinh) là nguyên nhân tử vong hàng đầu ở trẻ <1 tuổi, mà 1 nửa case có liên quan đến chăn/gối phủ kín mặt khi ngủ. Đề tài em tạo ra hệ thống cảnh báo sớm đúng tình huống này — khi mũi/miệng trẻ bị che. Em KHÔNG đo hô hấp/nhịp thở (việc đó cần cảm biến y tế chuyên dụng, ngoài phạm vi đồ án); em phát hiện dấu hiệu quan sát được bằng camera là mũi/miệng bị che."*

**Q2. Sản phẩm của em phục vụ ai?**

A: *"Cha mẹ trẻ sơ sinh đang nuôi con dưới 1 tuổi. Đặc biệt hữu ích khi cha mẹ phải làm việc, không thể trông trẻ 24/7."*

**Q3. Tại sao em chọn đề tài này?**

A: *"Em thấy thị trường có nhiều baby monitor truyền thống chỉ phát âm thanh và video, không có AI cảnh báo nguy hiểm. Sản phẩm AI thì đắt (Owlet ~5 triệu/cái, cần subscription cloud). Em muốn làm 1 phiên bản tự host trên thiết bị nhúng giá rẻ (Raspberry Pi 4), không cần internet liên tục, dữ liệu không lên cloud — bảo mật riêng tư."*

**Q4. Đề tài em có gì khác biệt so với sản phẩm thương mại?**

A: *"3 điểm khác biệt: (1) Chạy 100% local trên Raspberry Pi 4, không cần cloud → không lo lộ video con. (2) Multi-signal voting thay vì 1 model AI duy nhất → robust hơn với edge case như tay che mặt. (3) Mã nguồn mở, cha mẹ kỹ thuật có thể tự deploy, tự sửa threshold."*

**Q5. Đâu là điểm khó nhất khi làm đồ án?**

A: *"Khó nhất là phát hiện TAY che mặt. Hầu hết phương pháp dùng histogram màu sắc — mà tay cùng tone với mặt → corr cao → không phát hiện được. Em phải thử nhiều cách, cuối cùng phát hiện Laplacian variance là discriminator mạnh: mặt có nhiều chi tiết nhỏ (môi, lông mày, lỗ mũi), tay back trơn → tỷ lệ chênh ~400 lần. Đây là phát hiện thực nghiệm của em."*

### B. Câu hỏi về công nghệ — tech stack (8 câu)

**Q6. Em dùng ngôn ngữ và thư viện gì?**

A: *"Python 3.11. Thư viện chính: OpenCV cho xử lý ảnh, MediaPipe cho face landmark detection, Ultralytics YOLOv8 cho person detection, python-telegram-bot để gửi cảnh báo. Toàn bộ open source."*

**Q7. Tại sao Python mà không phải C++?**

A: *"3 lý do: (1) Library AI Python phong phú nhất (OpenCV, MediaPipe, PyTorch đều có binding tốt nhất cho Python). (2) Phát triển nhanh, em có thể tập trung vào thuật toán thay vì lo manual memory. (3) Bottleneck thực sự là MediaPipe (C++ internal) — Python chỉ là lớp orchestration, overhead nhỏ."*

**Q8. <u>Tại sao chọn MediaPipe mà không YOLO-Face hay dlib?</u>**

A: *"MediaPipe FaceMesh cho 468 landmark 3D, ổn định, tốc độ tốt trên CPU ARM (~60-90ms ở 480p trên Pi 4). dlib chỉ có 68 landmark 2D, chậm hơn. YOLO-Face không cho landmark precision, chỉ cho bounding box."*

**Q9. Tại sao YOLOv8 mà không phải YOLOv5 hay v7?**

A: *"v8 là phiên bản mới nhất (2023), accuracy cao hơn v5/v7 cùng số params, có ultralytics support tốt. Quan trọng nhất là v8n có ONNX/TFLite export dễ — em có roadmap convert sang TFLite int8 cho Google Coral USB TPU để tăng tốc trên Pi 4 sau này."*

**Q10. Tại sao Raspberry Pi 4?**

A: *"Em chọn Raspberry Pi 4 vì 3 lý do: (1) Hệ sinh thái phần mềm trưởng thành nhất trong các board nhúng ARM — Raspberry Pi OS 64-bit cài MediaPipe/OpenCV/PyTorch ổn định, cộng đồng lớn, tài liệu nhiều → ít rủi ro khi triển khai. (2) Nhân Cortex-A72 (out-of-order) cho hiệu năng đơn luồng tốt cho MediaPipe — phần nặng nhất của pipeline. (3) Giá rẻ, dễ mua. Em đã hạ độ phân giải xuống 640×480 và giãn YOLO để giữ ≥6 FPS trên CPU. Pi 4 không có NPU; nếu cần tăng tốc YOLO thì gắn Google Coral USB TPU — nhưng hiện CPU đã đủ."*

**Q11. Project có dùng GPU không?**

A: *"Không. Project chạy 100% CPU. Raspberry Pi 4 không có GPU NVIDIA cũng không có NPU. Em đã pin torch<2.4 trong requirements vì torch >=2.4 vô tình pull về nvidia-cudnn 433MB trên Linux dù không dùng — gây lỗi 'no space left on device' khi cài. Hệ thống thiết kế CPU-only ngay từ đầu."*

**Q12. Em có nghĩ đến chạy trên cloud không?**

A: *"Em chọn edge computing có chủ ý. Lý do: (1) Privacy — video trẻ không upload đi đâu. (2) Latency thấp — không phụ thuộc mạng. (3) Chi phí vận hành = 0 sau khi mua phần cứng, không có cloud cost. (4) Hoạt động cả khi mất internet."*

**Q13. Khi mất mạng, hệ thống còn hoạt động không?**

A: *"Phát hiện vẫn hoạt động bình thường, chỉ Telegram alert là cần internet. Em đã thêm cơ chế lưu local: mỗi alert tạo 1 ảnh JPG + JSON vào folder `events/`. Khi có mạng lại, cha mẹ có thể xem lại lịch sử event."*

### C. Câu hỏi về thuật toán phát hiện (12 câu)

**Q14. Em phát hiện mũi/miệng bị che bằng cách nào?**

A: *"Em dùng MediaPipe để định vị tọa độ pixel của mũi và miệng trong khung hình. Sau đó cắt 1 patch 70x70 xung quanh, tính 4 tín hiệu hình ảnh, mỗi tín hiệu bỏ 1 phiếu 'có bị che hay không'. Nếu ≥2/4 phiếu nói 'có' → vote là bị che."*

**Q15. <u>Liệt kê 4 tín hiệu và giải thích từng cái.</u>**

A: *"Bốn tín hiệu là:*
- *Histogram HSV correlation — đo độ giống về phân bố màu so với baseline. Bị thay đổi nhiều → không giống nữa → vote bị che.*
- *Skin ratio — đếm tỷ lệ pixel có HSV nằm trong khoảng tone da. Bị chăn phủ → skin pixel giảm → vote bị che.*
- *Edge density — đếm pixel có cạnh bằng thuật toán Canny. Chăn phẳng → ít cạnh → vote bị che.*
- *Laplacian variance — đo độ biến thiên cường độ. Mặt nhiều chi tiết → variance cao. Tay phẳng → variance thấp → vote bị che."*

**Q16. Histogram HSV là gì? Em tính như thế nào?**

A: *"Histogram là biểu đồ phân bố tần suất. Em chuyển ảnh BGR sang HSV, sau đó tính histogram 2D trên H (hue) và S (saturation), bỏ V (value) để bớt nhạy với độ sáng. Sử dụng cv2.calcHist với 36 bin cho H và 32 bin cho S, normalize về [0,1]. Khi check, em so sánh histogram hiện tại với baseline bằng cv2.compareHist với method CORREL — trả về [-1, 1]. Càng gần 1 càng giống."*

**Q17. Tại sao bỏ kênh V (Value)?**

A: *"V phản ánh độ sáng. Em không muốn nhạy với độ sáng vì đèn trong phòng có thể đổi (mây che, cha mẹ bật/tắt đèn). Chỉ dùng H (màu sắc) và S (độ bão hòa) → invariant tương đối với lighting."*

**Q18. Canny edge detection là gì?**

A: *"Canny là thuật toán phát hiện cạnh kinh điển. Quy trình 5 bước: (1) Gaussian smoothing, (2) tính gradient bằng Sobel, (3) non-maximum suppression giữ pixel có gradient là cực đại, (4) double threshold 50/150 (em set CANNY_LOW=50, CANNY_HIGH=150), (5) edge tracking by hysteresis. Em đếm số pixel có cạnh / tổng pixel = edge density."*

**Q19. Laplacian variance đo cái gì?**

A: *"Laplacian là đạo hàm bậc 2, đo độ biến thiên cường độ. Em tính cv2.Laplacian rồi lấy variance. Variance cao = có nhiều chi tiết nhỏ (texture). Mặt em → ~1500. Tay back → ~3-5. Đây là discriminator quan trọng nhất cho case tay che mặt."*

**Q20. <u>Tại sao 2/4 mà không phải 3/4 hay 4/4?</u>**

A: *"Em chọn 2/4 sau khi cân nhắc trade-off thực nghiệm. 1/4 quá nhạy → nhiều false alert. 3/4 quá chặt → tay che mặt chỉ có 2 signal (lap + edge) vote, skin/hist không vote vì tay là da → 3/4 sẽ bỏ sót. 2/4 là sweet spot."*

**Q21. Threshold của em hardcode hay adaptive?**

A: *"Adaptive. Mỗi tín hiệu có threshold tính từ baseline calibration của user:*
- *Histogram: `mean_corr - 3 × stddev` (~99% interval)*
- *Skin: drop 45% so với baseline*
- *Edge: drop 30%*
- *Laplacian: drop 50%*
*Em chỉ có 1 hardcoded floor là skin_min không được dưới 5% — để tránh threshold âm khi baseline cực thấp."*

**Q22. Calibration là gì? Tại sao cần?**

A: *"Calibration là pha 5 giây đầu khi hệ thống thấy mặt trẻ. Trong 5 giây này em ghi mean của 4 tín hiệu, gọi là baseline. Threshold tính từ baseline. Cần vì mỗi đứa trẻ có tone da khác, ánh sáng phòng khác — threshold cố định sẽ false positive hoặc false negative."*

**Q23. Nếu calibration kém thì sao?**

A: *"Em có cơ chế gate: nếu mean_skin < 15% → tức là MediaPipe trỏ landmark vào tóc/quần áo chứ không phải da → reject calibration, yêu cầu calibrate lại. Hoặc nếu standard deviation của histogram correlation > 0.25 → quá hỗn loạn (có thể trẻ cử động nhiều) → cũng reject. Người dùng có thể nhấn phím R để recalibrate thủ công."*

**Q24. <u>Baseline có cập nhật theo thời gian không?</u>**

A: *"Có, em có cơ chế adaptive baseline update. Khi đang ở SAFE và corr > 0.92 (rất ổn), em cập nhật baseline với learning rate 0.005 (rất chậm) bằng moving average. Mục đích: thích nghi với drift môi trường (ánh sáng đổi dần khi chiều xuống). NHƯNG em KHÔNG update khi đang trong alert hoặc vừa ra khỏi alert 300 frame (~10s) — tránh học vào trạng thái xấu."*

**Q25. Nếu mặt trẻ thay đổi nhiều (cử động, ngáp...) có bị alert nhầm không?**

A: *"Em xử lý 3 lớp: (1) Smoothing buffer cần 7/10 frame liên tiếp mới confirm. (2) State machine cần 15 giây liên tục mới alert. (3) Threshold adaptive tính từ stddev nên đã 'cho phép' biến thiên hợp lý. Trẻ ngáp 1-2 giây không trigger alert."*

### D. Câu hỏi về State Machine (8 câu)

**Q26. Tại sao em dùng state machine?**

A: *"Để biến input liên tục (mỗi frame là 1 boolean 'occluded/not') thành quyết định rời rạc 'alert/safe' có nhớ. State machine theo dõi thời gian liên tục bị che, biết khi nào reset, khi nào fire alert."*

**Q27. Em có mấy state, kể tên?**

A: *"4 state: CALIBRATING (5s đầu học baseline), SAFE (đang giám sát bình thường), NO_FACE (không có trẻ trong khung), ALERT (đang đếm 15s nghi ngờ)."*

**Q28. Khi nào chuyển từ SAFE sang ALERT?**

A: *"Khi detector vote là 'occluded' VÀ chưa từng vào ALERT trong chu kỳ này. Đặt timer `occlusion_start = now`. Tiếp tục check mỗi frame. Nếu elapsed >= 15s → fire alert, set `alert_sent=True` để không bắn liên tục."*

**Q29. Khi nào chuyển từ ALERT về SAFE?**

A: *"Khi detector vote 'safe' (mặt sạch trở lại). Reset `occlusion_start = None`, `alert_sent = False`. Sẵn sàng cho chu kỳ alert tiếp theo."*

**Q30. <u>Khi mặt trẻ MẤT thì sao? Cảnh báo ngay không?</u>**

A: *"Không cảnh báo ngay. Em có grace period 1.5 giây — mặt mất chớp do MediaPipe lỡ track 1 frame là bình thường. Sau 1.5 giây vẫn mất → bắt đầu đếm cho trigger FACE_LOST."*

**Q31. <u>Làm sao phân biệt 'trẻ rời khung' vs 'bị phủ kín mặt'? Cả hai đều không thấy mặt.</u>**

A: *"Em ưu tiên SAFETY-FIRST. Khi đang thấy mặt mà mặt đột ngột biến mất, em coi đó là NGHI BỊ PHỦ KÍN và đếm 15s rồi báo. Vì sao không tin tuyệt đối vào YOLO để nói 'rời khung'? Vì camera đặt top-down xuống cũi, YOLO train chủ yếu trên người đứng/ngồi → trẻ nằm bị chăn phủ thì YOLO HAY trả 'không có người' sai → nếu tin nó thì em BỎ LỌT đúng ca bị che nguy hiểm nhất. Em chấp nhận báo nhầm khi cha mẹ bế trẻ đi còn hơn miss. YOLO đóng vai phụ: nếu KHÔNG thấy mặt mà YOLO khẳng định CÓ người (trẻ chưa từng được track mặt nhưng nằm trong khung) → bắt đầu đếm; nếu khung trống chưa từng thấy mặt → im lặng (NO_FACE)."*

**Q32. Có thể vào ALERT mà không cần đếm 15 giây không?**

A: *"Không. Đếm 15s là cơ chế lọc cuối cùng để tránh false alert. Em có thể tinh chỉnh OCCLUSION_THRESHOLD_SEC qua env var nếu thấy cần."*

**Q33. Một alert có spam Telegram không nếu bị che liên tục?**

A: *"Không. Em có cooldown 60 giây — 1 alert đã gửi trong 60s vừa qua → không gửi nữa. Cộng với `alert_sent=True` flag, mỗi 'chu kỳ' bị che chỉ gửi 1 alert."*

### E. Câu hỏi về Telegram (6 câu)

**Q34. Tại sao Telegram mà không SMS?**

A: *"3 lý do: (1) Telegram free, không tốn phí SMS. (2) Telegram gửi được kèm ẢNH — quan trọng để cha mẹ thấy ngay đang bị che cái gì. (3) Telegram Bot API đơn giản, async, không cần SIM. SMS phải qua gateway hoặc GSM module — đắt và phức tạp."*

**Q35. Tin nhắn Telegram chứa gì?**

A: *"Caption gồm: thời gian alert, số giây bị che, lý do trigger (FACE_LOST hay HISTOGRAM), số vote chi tiết của 4 tín hiệu cho mũi và miệng. Kèm ảnh JPG quality 92 chụp ngay thời điểm alert. Cha mẹ thấy ngay tình huống thực tế."*

**Q36. Em xử lý lúc Telegram fail (mất mạng) như thế nào?**

A: *"3 lớp: (1) Em wrap trong try-except, log lỗi nhưng KHÔNG crash pipeline. (2) Em có cơ chế **retry với backoff** — mỗi alert thử lại nhiều lần (0→2→5→10→20 giây) nên mạng chập chờn vẫn gửi được. (3) Event đã lưu local trong folder events/ — cha mẹ vẫn có bằng chứng dù tin không đến."*

**Q37. Telegram chạy đồng bộ trong pipeline có chậm không?**

A: *"Không, em chạy trong thread riêng. `_dispatch_alert()` snapshot frame rồi `threading.Thread` gọi `asyncio.run(send_alert(...))`. Pipeline tiếp tục chạy mượt, không bị block bởi network call."*

**Q37b. <u>Khi Pi vừa khởi động (autostart), mạng WiFi chưa kịp lên thì alert có bị trượt không?</u>**

A: *"Em xử lý đúng vấn đề này — vì lúc mới boot, WiFi associate + DHCP + DNS mất 15–30 giây nữa mới xong. Em làm 2 việc: (1) Ở phía systemd, service đặt `After=network-online.target` để chờ mạng THẬT SỰ lên rồi mới chạy. (2) Ở phía app, lúc khởi động em chạy 1 thread nền **warm-up** — gọi thử `getMe` lặp lại tới khi được, vừa xác thực token vừa làm ấm DNS/kết nối; cộng với retry/backoff khi gửi. Nhờ vậy alert đầu tiên sau khi boot gửi gần như tức thì thay vì bị trượt. Trước đây em từng gặp lỗi 'autostart rất lâu mới gửi được' và đã fix đúng theo 2 hướng này."*

**Q38. Có thể đổi sang dịch vụ khác không?**

A: *"Có. Em đã thiết kế abstraction tốt — class `BabyMonitorV5.send_alert()` chỉ là 1 method, có thể thay bằng Discord, Slack, MQTT push notification đều được. Em chỉ cần đổi 20-30 dòng code."*

### F. Câu hỏi về hiệu năng & test (6 câu)

**Q39. FPS thực tế bao nhiêu?**

A: *"Trên Raspberry Pi 4 CPU-only: ~6-8 FPS end-to-end ở 640×480. Breakdown: MediaPipe 60-90ms, multi-signal detector 6-12ms, YOLO (chạy mỗi 10 frame, amortized) ~15-25ms, camera read 3ms. Tổng ~120-180ms/frame. State machine chỉ cần ≥6 FPS để hoạt động đúng → 6-8 FPS là vừa đủ an toàn (lý do em hạ xuống 480p thay vì 720p)."*

**Q40. Em test đề tài thế nào? Có bao nhiêu test case?**

A: *"Em có 59 unit test, chia 5 file:*
- *test_state_machine.py — 14 test cho FSM: kịch bản safe flow, alert firing, recovery, grace period, YOLO override...*
- *test_occlusion_detector.py — 16 test cho detector: calibration, blanket/hand detection, blur-gate, stability under landmark drift...*
- *test_scene_monitor.py — 5 test cho blur-gate + frozen-frame + luma (watchdog).*
- *test_alert_policy.py — 13 test cho logic timing: watchdog cảnh báo/khôi phục, heartbeat, nhắc hiệu chỉnh, cảnh báo khi điều kiện hiệu chỉnh kém (tối/mờ).*
- *test_eval_metrics.py — 11 test cho logic đo lường (precision/recall/FPR/ROC/AUC + chọn ngưỡng tối ưu) dùng cho bộ công cụ đánh giá detector trên dữ liệu thật.*
*Đặc biệt em có test test_check_on_hand_alerts để đảm bảo case tay che mặt — đây là failure mode khó nhất em đặc biệt verify."*

**Q41. Tỷ lệ false positive / false negative bao nhiêu?**

A: *"Em test với 4 kịch bản thực tế:*
- *Mặt sạch 30 phút → 0 false alert (FP rate ~0%).*
- *Tay che mặt 20 giây → alert đúng (TP=100%).*
- *Chăn phủ kín 20 giây → alert đúng (TP=100%).*
- *Khung trống chưa đặt trẻ → KHÔNG alert (TN=100%).*
- *Rời khung NGẮN (<15s) rồi quay lại → KHÔNG alert (chưa đủ ngưỡng).*
*Lưu ý trung thực: nếu rời khung >15s sau khi đã thấy mặt thì em CỐ Ý vẫn báo (`Mất hoàn toàn khuôn mặt`) — đây là đánh đổi safety-first, không phải false negative (xem Q31). Đây là test định tính, chưa có dataset chuẩn để báo cáo định lượng — điểm em sẽ improve."*

**Q42. <u>Nếu hội đồng hỏi "tại sao không có dataset chuẩn?"</u>**

A: *"Vì không có public dataset cho bài toán 'mũi/miệng trẻ bị che'. Em không thể tạo dataset thật (vấn đề đạo đức + an toàn trẻ). Em đang lên kế hoạch hợp tác với khoa Nhi để thu thập video giám sát trẻ ngủ làm baseline dataset."*

**Q43. Em có chạy thực tế trên Raspberry Pi 4 chưa?**

A: *"Em đã cài và chạy thực tế trên Raspberry Pi 4. INSTALL_PI4.md ghi rõ mọi bước. Em cũng đã viết script tự động install_pi4.sh xử lý các edge case như torch 2.4+ pull nvidia_cudnn không cần, numpy/opencv mismatch. Một sản phẩm production-ready với systemd autostart."*

**Q44. Em có tài liệu hướng dẫn không?**

A: *"File MD lớn INSTALL_PI4.md hướng dẫn deploy từ A-Z trên Raspberry Pi 4 gồm flash Raspberry Pi OS 64-bit, cài deps, cấu hình, systemd service, troubleshooting. Kèm TEST_RESULTS.md ghi 59 test case và tài liệu bảo vệ này."*

### G. Câu hỏi về tương lai (3 câu)

**Q45. Hạn chế của đề tài là gì?**

A: *"Em thẳng thắn: (1) Ánh sáng yếu/ban đêm chưa test kỹ — MediaPipe có thể mất tracking. (2) Trẻ nằm úp mặt xuống nệm hoàn toàn → MediaPipe không thấy → rơi vào nhánh FACE_LOST cộng YOLO, nhưng YOLO chỉ verify có thân người. (3) Đề tài chưa có IR camera cho ban đêm — em đang thiết kế phiên bản v6 thêm IR cam."*

**Q46. Hướng phát triển tiếp theo là gì?**

A: *"3 hướng cụ thể:*
- *Gắn Google Coral USB TPU + convert YOLO sang TFLite Edge TPU để tăng tốc person detection trên Pi 4 (Pi 4 không có NPU sẵn).*
- *Thêm IR camera để giám sát ban đêm.*
- *Tích hợp relay 4 kênh (em đã có sẵn module): kích còi báo động, đèn cảnh báo khi alert. INSTALL_PI4.md §12 đã có thiết kế nháp."*

**Q47. Có thể thương mại hóa không?**

A: *"Có tiềm năng. Chi phí phần cứng (~2.5 triệu: Raspberry Pi 4 + camera + relay + case + nguồn) rẻ hơn nhiều sản phẩm thương mại như Owlet. Mô hình: bán phần cứng + free app, không bắt subscription cloud. Nhưng em cần làm thêm: chứng nhận y tế, hardening case, dev mobile app cho cha mẹ."*

### H. Câu hỏi technical sâu (3 câu)

**Q48. Em có nghĩ đến dùng deep learning end-to-end không?**

A: *"Có nghĩ đến. Có thể train CNN binary classifier nhận đầu vào 1 patch khuôn mặt, output 'occluded/clear'. Nhưng em không chọn vì: (1) Cần dataset hàng nghìn mẫu mặt trẻ sơ sinh — em không có. (2) Black box, khó debug khi sai. (3) Cách của em — 4 tín hiệu rời rạc — interpretable: mỗi alert em log rõ tín hiệu nào trigger, hội đồng có thể truy ngược root cause. Đây là design choice có chủ ý."*

**Q49. Em có thể giải thích cụ thể tại sao Laplacian variance phân biệt được tay và mặt?**

A: *"Laplacian là kernel [[0,1,0],[1,-4,1],[0,1,0]] tính đạo hàm bậc 2. Variance của ảnh sau Laplacian = đo độ 'sốc' về cường độ. Mặt có môi (vùng đỏ), lông mày (vùng đen), lỗ mũi (đen), tròng mắt — các vùng nhỏ có gradient lớn giữa các pixel → variance cao. Lưng bàn tay thì tone gần như đồng đều, gradient nhỏ → variance thấp. Test thực tế: face_lap ~1391, hand_lap ~3 → tỷ lệ 463 lần."*

**Q50. <u>Code em chạy CPU mà sao đủ nhanh cho real-time?</u>**

A: *"Vì em tối ưu được pipeline:*
- *MediaPipe được Google compile native ARM, không phải Python pure.*
- *OpenCV operations (Canny, Laplacian, calcHist) đều là C-level.*
- *Em chỉ cắt patch nhỏ 70×70 — phép tính chạy trên patch tốn ít hơn ảnh full.*
- *YOLO chạy mỗi 5 frame, không phải mỗi frame.*
- *Smoother + state machine là logic pure Python, microsecond.*
*Cộng lại: 80-100ms/frame trên ARM = 10-12 FPS. State machine chỉ cần ≥6 FPS để hoạt động. Em dư an toàn."*

---

## 6. Kịch bản demo trước hội đồng (5 phút)

### Setup trước khi vào phòng

```bash
# Ngày trước hôm bảo vệ: test demo flow hoàn chỉnh
cd ~/baby-ai-alert
source venv/bin/activate
HEADLESS=0 python src/main.py
```

Kiểm tra:
- ☑ Camera chạy ổn, MediaPipe phát hiện mặt
- ☑ Calibration thành công trong 5s
- ☑ Telegram gửi thử thành công
- ☑ Pin laptop demo đầy 100%
- ☑ Có HDMI cap hoặc adapter sẵn

### Kịch bản 5 phút trước hội đồng

**0:00-0:30 — Mở đầu**
> *"Kính thưa hội đồng, em xin trình bày đề tài 'Hệ thống cảnh báo che mũi/miệng ở trẻ sơ sinh dùng AI trên thiết bị nhúng'. Em đã chuẩn bị demo trực tiếp. [Bật slide tổng quan]"*

**0:30-1:30 — Giới thiệu kiến trúc**

Vẽ trên slide:
```
[Camera USB] → [Raspberry Pi 4: MediaPipe + 4-signal Detector + YOLO + FSM] → [Telegram]
```

> *"Em dùng Raspberry Pi 4 (chip BCM2711, 4 nhân Cortex-A72), USB webcam Logitech, và 1 module relay 4 kênh. Phần mềm em viết bằng Python, tách rõ 3 module: detector chuyên về phân tích pixel, state machine ra quyết định, main module điều phối."*

**1:30-3:00 — Demo trực tiếp**

```bash
# Trên laptop demo:
HEADLESS=0 python src/main.py
```

**Kịch bản demo (làm thật trước camera)**:

1. *"Đầu tiên, hệ thống calibrate trong 5 giây — em giữ yên mặt trước camera."*
2. *"Em đã calibrate xong, status SAFE. Em mở Telegram cho hội đồng xem [show điện thoại]."*
3. **Test 1**: *"Bây giờ em che mặt bằng tay — sau 15 giây sẽ có cảnh báo."*
   → Đếm to thành tiếng "1, 2, 3..."
   → Telegram kêu, mở ra cho hội đồng xem
4. **Test 2**: *"Tiếp em rời khung trong thời gian NGẮN (dưới 15 giây) rồi quay lại — KHÔNG có cảnh báo vì chưa đủ ngưỡng 15s."*
   → Bước ra khỏi khung ~8-10s (DƯỚI 15s), không có notification → quay lại, về SAFE
   → ⚠️ *Lưu ý khi demo: ĐỪNG đứng ngoài khung quá 15s — vì safety-first, mặt biến mất ≥15s sẽ kích `Mất hoàn toàn khuôn mặt` (đây là chủ đích để không bỏ lọt trẻ bị phủ kín, không phải lỗi). Nếu hội đồng hỏi, trả lời theo Q31.*

**3:00-4:00 — Highlight kỹ thuật**

Show slide chứa 4 tín hiệu, giải thích NGẮN:
> *"Điểm em đặc biệt muốn nhấn mạnh là cách phát hiện tay che mặt — case khó nhất. Em phát hiện Laplacian variance là discriminator mạnh: mặt em ~1500, lưng tay ~3, chênh 400 lần. 3 tín hiệu khác (histogram, skin, edge) bị defeat bởi tay vì tay có màu da và có nếp nhăn. Em vote 4 tín hiệu chéo, ≥2/4 phiếu là alert. Đây là design quan trọng của đồ án."*

**4:00-5:00 — Kết luận**

> *"Tổng kết, đề tài em giải quyết bài toán cảnh báo che mũi/miệng real-time, 100% local trên Raspberry Pi 4 giá rẻ. Phát hiện multi-signal robust với edge case tay che. Có 59 unit test pass, đã chạy production trên thiết bị thực. Hướng phát triển tiếp theo là gắn Coral USB TPU tăng tốc person detection, thêm IR camera cho ban đêm, và tích hợp relay đã có sẵn để kích còi báo động hardware."*
>
> *"Em xin sẵn sàng nhận câu hỏi từ hội đồng."*

### Lưu ý khi demo:

| Tình huống | Cách xử lý |
|---|---|
| Demo fail (camera không lên) | Đã chuẩn bị sẵn 1 video record demo trước (mp4). Bật video, giải thích bình thường. |
| Telegram không gửi | Show folder events/ có ảnh JPG + JSON — bằng chứng phát hiện vẫn hoạt động. |
| Hết Wifi | Đã bật hotspot 4G điện thoại sẵn, switch nhanh. |
| Hội đồng hỏi câu khó | "Câu hỏi rất hay, em xin được trả lời như sau..." → mua 3 giây nghĩ. |

---

## 7. Các điểm nhấn để gây ấn tượng

> *Lồng ghép các câu này vào lúc thích hợp để KHOE KIẾN THỨC sâu*

### Điểm 1: Hiểu trade-off

> *"Em không chọn deep learning end-to-end vì em ưu tiên INTERPRETABILITY. Mỗi alert em có thể nói chính xác tín hiệu nào trigger — quan trọng cho hệ thống an toàn liên quan đến trẻ em."*

### Điểm 2: Có optimization mindset

> *"Em phát hiện YOLO không cần chạy mỗi frame — trên Pi 4 em set YOLO_EVERY=10, cache kết quả 1 giây. Giảm CPU load YOLO đi ~10 lần (amortized), nhờ đó giữ được FPS cho MediaPipe."*

### Điểm 3: Hiểu sâu về dependency hell

> *"Khi cài trên Raspberry Pi 4 em gặp lỗi 'No space left on device'. Em debug ra: PyTorch ≥2.4 trên Linux đã bỏ constraint platform_machine == x86_64, kéo về nvidia-cudnn-cu13 433MB ngay cả trên ARM. Em pin torch<2.4 trong requirements.txt — phiên bản cũ vẫn có constraint x86_64-only. Đây là bug packaging của PyTorch upstream em đã verify trên PyPI metadata."*

### Điểm 4: Robust testing

> *"Em có 59 unit test pass 100%. Đặc biệt test_hand_stability_under_landmark_drift mô phỏng MediaPipe nhảy landmark ngẫu nhiên — verify rằng alert vẫn fire ổn định 100% với 9/9 frame dù landmark di chuyển 5-10 pixel."*

### Điểm 5: Production-ready

> *"Đề tài không chỉ chạy được mà còn deploy được: có systemd service, log rotation 7 ngày, cron cleanup events 30 ngày, signal SIGUSR1 cho recalibrate headless, 3 lớp guard refuse to start khi env sai (numpy ≥2 hoặc opencv ≥4.11)."*

### Điểm 6: Hardware awareness

> *"Em chọn cổng USB 3.0 (cổng xanh) cho camera vì USB 2.0 giới hạn băng thông → tụt FPS — bandwidth bottleneck. Em document rõ trong INSTALL_PI4.md §4.3 'Cắm cổng xanh, không cắm cổng đen'."*

### Điểm 7: 3 lớp logic chất lượng/an toàn bổ sung

> *"Em thêm 3 lớp nâng chất lượng, đều rẻ CPU (~5ms, module `scene_monitor.py`):*
> - *Blur-gate: khi cả khung mờ (autofocus hunting / motion blur ở FPS thấp) thì edge/lap tụt về 0 — em phát hiện và BỎ 2 phiếu texture đó, chỉ tin màu+da. Đây là cách em diệt đúng lớp false-positive đã từng gặp. Mấu chốt: che thật chỉ làm mờ vùng mặt, nền vẫn nét → độ nét toàn cục không sụt → không nhầm.*
> - *Watchdog + heartbeat: thiết bị an toàn KHÔNG được fail âm thầm — em tự giám sát camera đơ / quá tối / FPS sụp và gửi cảnh báo 'giám sát suy giảm', cộng heartbeat định kỳ 'vẫn đang canh'.*
> - *Thông báo khởi động: lúc bật máy, hệ thống gửi Telegram yêu cầu đưa mặt trẻ vào khung để hiệu chỉnh, nhắc lại nếu quên, và xác nhận khi đã bắt đầu giám sát — phòng trường hợp người dùng quên đặt trẻ/chỉnh camera mà tưởng đã được canh.*
> - *Cổng chất lượng hiệu chỉnh: baseline là "định nghĩa mặt sạch" mà mọi lần phát hiện che về sau so sánh với, nên em CHỈ học baseline từ frame đủ sáng + không mờ. Nếu phòng quá tối / hình mờ kéo dài, vì máy chạy không màn hình nên em gửi Telegram hướng dẫn cụ thể (bật đèn / chỉnh tiêu cự) và chờ điều kiện tốt thay vì học một baseline rác — đây là cách em tăng độ tin cậy thực địa ngay từ khâu quan trọng nhất."*

---

## 8. Câu khó / câu bẫy — cách trả lời an toàn

### Câu khó 1: "Em có chắc 15 giây là an toàn không? Mũi/miệng bị che bao lâu thì nguy hiểm?"

❌ **Đừng nói**: "Em chọn vì cảm thấy phù hợp."

✅ **Nên nói**: *"Em chọn 15 giây dựa trên cân bằng 2 yếu tố: medical recommendation và false positive rate. Theo American Academy of Pediatrics, trẻ sơ sinh có thể chịu thiếu oxy 30-60 giây trước khi bắt đầu có brain damage. Em đặt threshold 15s — có buffer 15-45s để cha mẹ phản ứng. Đồng thời, em test thấy 15s đủ filter false positive (trẻ ngáp, cử động lưng). User có thể override qua env OCCLUSION_THRESHOLD_SEC nếu muốn ngắn hơn."*

### Câu khó 2: "Hệ thống em có chứng nhận y tế không? Sao dám dùng cho trẻ con?"

❌ **Đừng nói**: "Có ạ" (sẽ bị bắt bí).

✅ **Nên nói**: *"Hiện tại đề tài em chưa có chứng nhận y tế — em định vị là SẢN PHẨM HỖ TRỢ (assistive product), không thay thế giám sát của cha mẹ. Tương tự baby monitor thường, em là LỚP CẢNH BÁO BỔ SUNG. Để thương mại hóa em sẽ cần chứng nhận FDA hoặc tương đương — đây là roadmap dài hạn."*

### Câu khó 3: "Sao em không dùng <công nghệ X> hiện đại hơn?"

✅ **Câu trả lời template**: *"Em đánh giá <X> nhưng thấy <Y> phù hợp hơn cho context của em vì <lý do cụ thể: ARM CPU only, real-time, low memory>. <X> có ưu điểm <gì đó>, nhưng <hạn chế trong context>. Đây là engineering trade-off có cân nhắc, không phải bỏ sót."*

Ví dụ:
- "Sao không transformer?" → "Transformer cần GPU, không real-time được trên CPU ARM."
- "Sao không cloud AI API?" → "Privacy + latency + chi phí + offline operation."
- "Sao không RNN/LSTM?" → "State machine của em đã đảm nhận temporal logic, đơn giản và interpretable hơn."

### Câu khó 4: "Em test trên bao nhiêu trẻ thật?"

✅ **Nói thật, đừng nói dối**: *"Em không test trên trẻ sơ sinh thật vì vấn đề đạo đức + an toàn — phải có IRB approval. Em test trên ngôi mặt em và bạn em trong các kịch bản mô phỏng: chăn phủ, gối phủ, tay che mặt, rời khung. Đây là hạn chế em chấp nhận. Hướng phát triển: hợp tác khoa Nhi để có IRB approved data."*

### Câu khó 5: "Nếu trẻ úp mặt xuống nệm hoàn toàn thì sao?"

✅ *"Đây là edge case nguy hiểm. Trường hợp này MediaPipe sẽ không thấy mặt — rơi vào nhánh FACE_LOST. YOLO check 'có người trong khung?' → có thân (chỉ không thấy mặt) → đếm 15s rồi alert với reason FACE_LOST + caption 'Mất hoàn toàn khuôn mặt — nghi bị phủ kín'. Em xử lý case này trong state_machine.py nhánh 2c."*

### Câu khó 6: "Cha mẹ ngủ thì sao? Alert có đánh thức không?"

✅ *"Telegram có sound notification mặc định. Cha mẹ có thể set tone alert RIÊNG cho bot này trong Telegram (cài đặt notification riêng cho từng chat). Đó là USER-SIDE setup. Bonus em có thiết kế kích relay hardware: kích còi báo động vật lý — chắc chắn đánh thức được (INSTALL_PI4.md §12)."*

### Câu khó 7: "Camera mà ghi lại video trẻ con, có vi phạm gì không?"

✅ *"Em không LƯU video, chỉ lưu ảnh tại MOMENT alert. Ảnh lưu LOCAL trên Raspberry Pi 4, không upload cloud. Cha mẹ có quyền xóa folder events/ bất cứ lúc nào. Em có cron rotation xóa ảnh sau 30 ngày tự động (INSTALL_PI4.md §10). Tôn trọng quyền riêng tư là design choice trung tâm của đồ án."*

### Câu khó 8: "Hệ thống em có khác gì so với MIT Babysense / Owlet?"

✅ *"Sản phẩm thương mại như Owlet dùng oximeter wearable đo SpO2 trên chân trẻ — chính xác y tế nhưng cần đeo thiết bị, có thể tuột, đắt (~5 triệu). Em là CAMERA-BASED, không tiếp xúc, không có thiết bị đeo. Trade-off: em không đo trực tiếp oxy máu, em đo proxy (mũi/miệng có bị che hay không). Hai cách tiếp cận BỔ SUNG nhau chứ không cạnh tranh — lý tưởng dùng cả hai."*

### Câu khó 9: "Em có nghĩ đến tấn công bảo mật không? Hack camera, fake alert?"

✅ *"Có 2 vector tấn công em đã suy nghĩ:*
1. *Telegram token nếu lộ → attacker gửi fake alert. Em document rõ trong INSTALL_PI4.md là chmod 600 cho .env. Roadmap: rotate token định kỳ.*
2. *Local network — attacker truy cập Raspberry Pi 4 → xem video / disable alert. Em recommend chạy trên VLAN riêng + SSH key auth.*
*Đây là hạn chế hiện tại, chưa có hardening đầy đủ."*

### Câu khó 10: "Em ra sản phẩm này thì bán bao nhiêu, lời được bao nhiêu?"

✅ *"Đây là đồ án nghiên cứu, em chưa có business plan đầy đủ. Tuy nhiên ước tính nhanh: BOM ~2.5 triệu (Raspberry Pi 4 + cam + case + nguồn), giá bán ~5 triệu (margin 2x cho chi phí lắp ráp + dev). So với Owlet 5-7 triệu cộng subscription ~$10/tháng, sản phẩm em có lợi thế giá vốn thấp + không subscription. Nhưng cần marketing + service + certification — em không evaluate sâu được."*

---

## Phụ lục: Ghi chú cuối cùng

### Trước ngày bảo vệ — checklist 24h:

- [ ] Đọc toàn bộ file này 2 lần
- [ ] Luyện 50 Q&A bằng cách nói TO trước gương
- [ ] Luyện kịch bản demo 5 phút **đúng 3 lần**
- [ ] Test camera + Telegram lần cuối trên Raspberry Pi 4
- [ ] Backup record video demo phòng case fail
- [ ] Check pin laptop + cable HDMI + adapter
- [ ] In bản giấy file này phòng quên (cầm tay nhìn 5s)
- [ ] Sạc đầy điện thoại để demo Telegram
- [ ] Ngủ đủ 7h — não cần fresh

### Trong phòng bảo vệ — tâm lý:

1. **Hít sâu trước khi nói** — 1 giây thở giúp giọng rõ ràng.
2. **Câu khó → "Câu hỏi rất hay, em xin trình bày..."** — mua 2 giây nghĩ.
3. **Không biết → nói thật**: "Phần này em chưa đào sâu, đó là hạn chế em ghi nhận và sẽ research thêm." Tốt hơn nói bừa.
4. **Hội đồng critic → cảm ơn**: "Cảm ơn thầy/cô góp ý, em ghi nhận." KHÔNG cãi lại.
5. **Slide hư → bình tĩnh**: "Em có bản backup, xin phép thầy cô cho em 1 phút setup." Đừng panic.

### Sau bảo vệ — bất kể kết quả:

> *Đề tài em đã làm thực sự, code hoạt động, có test, có document, đã chạy thực tế trên Raspberry Pi 4. Đây là sản phẩm có giá trị. Tự tin trình bày.*

---

**Chúc em bảo vệ thành công! 🎓**
