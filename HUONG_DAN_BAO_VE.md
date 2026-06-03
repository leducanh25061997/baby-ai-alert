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

**"Đề tài của em là hệ thống cảnh báo cho cha mẹ trẻ sơ sinh bằng AI, chạy trên máy tính nhúng Raspberry Pi 4. Hệ thống dùng camera quan sát trẻ liên tục, dùng MediaPipe để xác định vị trí mũi và miệng, sau đó kiểm tra xem vùng mũi/miệng có còn là màu da hay đã bị vật lạ (chăn/gối/khăn/giấy) che. Hệ thống chỉ gửi 2 loại thông báo: (1) khi mũi/miệng bị che liên tục quá 15 giây, và (2) khi không thấy ai trong khung quá 15 giây. Mỗi thông báo gửi về Telegram cho cha mẹ kèm ảnh chụp tại thời điểm đó. Toàn bộ chạy CPU-only trên Raspberry Pi 4, không cần GPU, hoạt động real-time khoảng 6-8 khung hình mỗi giây ở độ phân giải 640×480."**

### 4 điểm mạnh để khoe ngay:
1. **Real-time trên thiết bị nhúng giá rẻ** (Raspberry Pi 4 ~1.5 triệu VND) — không cần cloud, không cần GPU
2. **Tín hiệu phát hiện đơn giản và bền** — quyết định "bị che" chỉ dựa vào tỷ lệ màu da (skin ratio) tụt mạnh → mặt nhìn rõ KHÔNG bao giờ bị báo nhầm
3. **Có state machine + anti-flicker** — tránh false alert khi mặt bị che hay rớt track trong tích tắc
4. **Có người = MediaPipe HOẶC YOLO** — giữ thêm 2 giây sau lần thấy gần nhất nên mặt nhìn rõ không bị hiểu nhầm thành "mất người"; chỉ báo "mất người" khi thật sự không còn ai trong khung

---

## 2. Cấu trúc project — sơ đồ nhà

Hình dung project như **một ngôi nhà 3 phòng**:

```
baby-ai-alert/
│
├── src/                              ← "Khu vực chính" — code chạy app
│   ├── main.py                       ← Phòng khách: tiếp nhận camera, điều phối
│   ├── state_machine.py              ← Phòng não: quyết định khi nào báo che / mất người
│   ├── occlusion_detector.py         ← Phòng mắt: nhìn và phân tích pixel mũi/miệng
│   ├── scene_monitor.py              ← Phòng cảm biến: blur-gate + motion + luma
│   └── alert_policy.py               ← Phòng đồng hồ: timing watchdog/heartbeat/nhắc hiệu chỉnh
│
├── tests/                            ← Kho kiểm thử (52 unit test)
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
| **state_machine.py** | Não bộ ra quyết định: SAFE / COVERED / NO_PERSON | ~150 |
| **occlusion_detector.py** | Mắt: phân tích pixel ở mũi/miệng dựa vào skin-ratio để biết "có bị che không" | ~320 |

### Tại sao chia 3 file mà không gộp 1?

**Câu trả lời chuẩn**: *"Em chia theo nguyên tắc Separation of Concerns. State machine không cần biết về OpenCV hay MediaPipe, nó chỉ nhận input boolean và quyết định state. Detector không cần biết về Telegram hay camera. Tách ra giúp em viết unit test cho state machine mà không cần camera thật — em có 52 test pass 100%. Nếu mai sau đổi từ MediaPipe sang model khác, em chỉ cần thay detector, state machine không phải đụng."*

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
  ┌────────┴────────┐
  │ YOLO check      │  ~40-80ms(CPU), chạy mỗi 10 frame
  │ "có người       │
  │  trong khung?"  │
  └────────┬────────┘
           │
    ┌──────┴──────┐
    │             │
  CÓ MẶT      KHÔNG MẶT
    │             │
    ▼             │
┌─────────────┐   │
│ Detector    │   │  CÓ NGƯỜI = thấy mặt HOẶC YOLO thấy người
│ skin-ratio  │   │  (giữ thêm 2s sau lần thấy gần nhất)
│ ~6-12ms     │   │
└──────┬──────┘   │
       │          │
       └────┬─────┘
            ▼
   ┌─────────────────┐
   │ State Machine   │  ← QUYẾT ĐỊNH: SAFE / COVERED / NO_PERSON
   │  + Smoother     │     dựa trên thời gian liên tục
   └────────┬────────┘
            ▼
    ┌──────────────────┐
    │ CẦN THÔNG BÁO?   │
    └──┬────────────┬──┘
       │            │
     Không         Có
       │            │
       │            ▼
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
3. **YOLO check người** (mỗi 10 frame): xác định trong khung còn ai không. Kết hợp với MediaPipe để biết "có người": thấy mặt HOẶC YOLO thấy người. Sau lần thấy gần nhất còn giữ "có người" thêm `PRESENCE_HOLD_SEC` (mặc định 2s) để MediaPipe rớt track vài frame trên Pi 4 không bị hiểu thành "mất người".
4. **Cắt patch quanh landmark** (~1ms): mỗi điểm landmark, em cắt 1 ô vuông 70×70 pixel xung quanh. Đó là khu vực mũi/miệng cần giám sát.
5. **Tính tín hiệu cho mỗi patch** (~8ms):
   - Tỷ lệ pixel da (skin ratio) — tín hiệu QUYẾT ĐỊNH
   - Histogram HSV (phân bố màu) — chỉ để hiển thị/log
6. **Quyết định "bị che"**: nếu vùng mũi HOẶC miệng có skin ratio tụt sâu (mất màu da → có vật che) thì coi là bị che. Vote hiển thị dạng `/2` (skin + hist) nhưng chỉ skin quyết định.
7. **Smoother** (lọc nhiễu): cần ≥7/10 frame liên tiếp nói "bị che" mới chính thức coi là bị che → tránh false alert do landmark nhảy 1 frame.
8. **State machine**: nếu mũi/miệng bị che liên tục 15 giây → báo **CHE MŨI/MIỆNG**; nếu không thấy ai trong khung liên tục 15 giây → báo **MẤT NGƯỜI**.
9. **Gửi thông báo**:
   - Lưu ảnh JPG + JSON metadata vào `events/`
   - Gửi ảnh + caption qua Telegram Bot API (chạy ở thread riêng để không chặn pipeline)

### Em chỉ cần thuộc 5 con số:

| Thông số | Giá trị | Ý nghĩa |
|---|---|---|
| **FPS pipeline** | 6-8 FPS | Tốc độ thực tế trên Raspberry Pi 4 (640×480) |
| **Ngưỡng báo che** | 15 giây | Mũi/miệng bị che liên tục bao lâu thì báo (`OCCLUSION_THRESHOLD_SEC`) |
| **Ngưỡng mất người** | 15 giây | Không thấy ai bao lâu thì báo (`NO_PERSON_SEC`) |
| **Giữ presence** | 2 giây | Giữ "có người" sau lần thấy gần nhất (`PRESENCE_HOLD_SEC`) |
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

### 4.2 Tại sao quyết định "bị che" CHỈ dựa vào skin-ratio?

**Đây là điểm THIẾT KẾ QUAN TRỌNG sau khi viết lại logic. Học thuộc luôn:**

> *"Em rút gọn lại: chỉ dùng tỷ lệ màu da (skin ratio) để quyết định. Lý do: mặt sạch — dù sáng, mờ hay áp sát camera — vùng mũi/miệng LUÔN là màu da. Vật che thật (chăn, gối, khăn, giấy, đồ chơi) thì KHÔNG phải màu da → skin ratio tụt mạnh. Đây là tín hiệu bền nhất với mặt sạch nên gần như không báo nhầm."*

| Tín hiệu | Vai trò | Lý do |
|---|---|---|
| **Skin ratio (tỷ lệ da)** | QUYẾT ĐỊNH | Chăn/gối/khăn/giấy không có pixel da → skin tụt → coi là che. Mặt sạch luôn nhiều da → không bao giờ kích. |
| **Histogram (màu sắc)** | Chỉ hiển thị/log | Vùng miệng cử động (bé mấp máy môi) làm histogram tụt oan → KHÔNG dùng để quyết định, chỉ để xem/ghi log. |

**Tại sao bỏ edge density và Laplacian variance?**
> *"Hai tín hiệu cũ đó đo ĐỘ NÉT của ảnh chứ không đo có vật che. Trên mặt nhìn rõ nhưng hơi mờ / áp sát camera, edge và lap_var tụt về gần 0 → bị hiểu nhầm là 'bị che' dù mặt vẫn sạch (xem các event cũ: skin=1.0, hist=0.96 mà vẫn bị vote che chỉ vì edge=0). Em đã bỏ hẳn 2 tín hiệu này để diệt đúng lớp false-positive đó."*

### 4.3 Quyết định "bị che" — chỉ skin quyết định

```
Patch mũi:  skin ratio, hist (hiển thị /2 phiếu)
Patch miệng: skin ratio, hist (hiển thị /2 phiếu)

Quyết định: BẤT KỲ patch nào (mũi HOẶC miệng) có skin ratio tụt
            dưới ngưỡng (mất màu da) → coi là bị che.
            Histogram chỉ hiển thị, KHÔNG tham gia quyết định.
```

**Ngưỡng skin tính sao?**
> *"Sau calibration, em tính ngưỡng `skin_min = skin_baseline × 0.5` (báo khi skin tụt hơn 50% so với baseline), kèm sàn tuyệt đối 0.15. Vật che thật làm skin tụt mạnh, vượt qua ngưỡng này; mặt sạch dao động nhẹ thì không."*

**Tại sao BẤT KỲ patch nào, không phải CẢ HAI?**
> *"Vì trong tình huống thực tế, đôi khi miệng bị che mà mũi vẫn lộ một phần, hoặc ngược lại. Quan trọng là EM BÁO KỊP THỜI — an toàn của trẻ. Em chọn: 1 trong 2 vùng mất màu da → báo che."*

### 4.4 Calibration — "học baseline khuôn mặt"

**Vấn đề**: Mỗi đứa trẻ có tone da khác nhau, ánh sáng phòng khác nhau. Threshold cố định không hoạt động.

**Giải pháp**: Trong 5 giây đầu, app GHI NHỚ baseline mặt sạch:
- Trung bình skin ratio (mean skin)
- Trung bình histogram (mean color distribution) — chỉ để tham chiếu/hiển thị

Sau đó tính threshold **ADAPTIVE** dựa trên phân bố quan sát:
- `skin_min = mean_skin × 0.5` (báo khi giảm hơn 50%), kèm sàn tuyệt đối 0.15
- `hist_threshold` được tính nhưng KẸP bảo thủ trong [0.30, 0.55] và chỉ dùng để hiển thị/log

**Câu mẫu giải thích**:
> *"Em không hardcode threshold mà tính từ data thực tế của từng phiên. Mỗi user có baseline riêng. Tín hiệu quyết định là skin ratio — ngưỡng tính từ skin trung bình của các sample calibration."*

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

**3 trạng thái cấp UI** (cộng pha CALIBRATING khi mới khởi động):

| Trạng thái | Khi nào | Hành động |
|---|---|---|
| **CALIBRATING** | 5 giây đầu khi thấy mặt | Học baseline |
| **NO_PERSON** | Không thấy mặt VÀ YOLO cũng không thấy người (đã quá thời gian giữ presence) | Đếm 15s → báo "Không thấy ai trong khung" |
| **SAFE** | Có người, vùng mũi/miệng còn màu da | Giám sát bình thường |
| **COVERED** | Có người, vùng mũi/miệng mất màu da (đang đếm/đã báo) | Đếm 15s → báo "Mũi/miệng bị che" |

**2 trigger dẫn đến thông báo**:
1. `covered` (`TRIGGER_COVERED`) — có người + vùng mũi/miệng bị che liên tục ≥ 15s.
2. `no_person` (`TRIGGER_NO_PERSON`) — không thấy ai trong khung liên tục ≥ 15s.

> *Không còn khái niệm "ngạt thở / nghi bị phủ kín". "Mất người" chỉ là thông báo sự việc ("không thấy ai trong khung"), KHÔNG suy diễn nguy hiểm.*

**Câu thoại giải thích**:
> *"Em không gửi thông báo ngay khi phát hiện che — có thể là che chớp 1 giây không sao. Em đếm 15 giây liên tục. Có anti-flicker: phải sạch trở lại liên tục đủ lâu (1.5s) mới reset bộ đếm, nên 1 frame nhiễu không làm reset oan. Tương tự cho 'mất người'. Sau lần đầu đủ ngưỡng, nếu tình huống còn kéo dài thì nhắc lại định kỳ đến khi hết."*

### 4.7 YOLO — phân biệt "còn người" vs "không còn ai"

**Tại sao có YOLO?**

> *"YOLO không phát hiện bị che — đó là việc của detector chính. Vai trò của YOLO là giúp xác định CÓ NGƯỜI trong khung hay không. 'Có người' = MediaPipe thấy mặt HOẶC YOLO thấy người. Nhờ YOLO, khi bé quay đầu / úp mặt (MediaPipe không thấy mặt nhưng YOLO vẫn thấy thân người) thì hệ thống biết VẪN CÒN người trong khung → KHÔNG báo nhầm 'mất người', và cũng không tạo cảnh báo che mới vì không quan sát được mũi/miệng.*
> *Ngược lại, khi thật sự không còn ai (cả MediaPipe lẫn YOLO đều không thấy) liên tục quá 15s → báo 'Không thấy ai trong khung'."*
>
> *"Để mặt nhìn rõ KHÔNG bao giờ bị báo nhầm 'mất người', em giữ thêm `PRESENCE_HOLD_SEC` (mặc định 2s) sau lần thấy gần nhất — MediaPipe trên Pi 4 đôi khi rớt track vài frame, nhưng trong 2s đó vẫn coi là 'có người'."*

**Tại sao YOLOv8n (nano) mà không phải s/m/l?**
> *"Nano là phiên bản nhỏ nhất, 6MB, nhẹ nhất cho CPU. s/m/l chậm hơn nhiều, không real-time được trên Raspberry Pi 4 (không có NPU). Em không cần độ chính xác cao, chỉ cần biết 'có thân người' hay không."*

**Tại sao không chạy YOLO mỗi frame?**
> *"Để tối ưu CPU. Trên Pi 4 em set `YOLO_EVERY=10` — chỉ chạy 1 lần mỗi 10 frame, cache kết quả. Lý do: xác định 'còn người trong khung hay không' không cần real-time — 1-2 lần/giây là dư."*

---

## 5. Bộ câu hỏi — câu trả lời chuẩn (50 câu)

> *Học thuộc các câu in **đậm**. Câu khó có gạch dưới — luyện trả lời ngắn gọn, đừng vòng vo.*

### A. Câu hỏi về mục đích đề tài (5 câu)

**Q1. Em làm đề tài này giải quyết bài toán gì?**

A: *"Em giải quyết bài toán phát hiện sớm khi mũi/miệng trẻ sơ sinh bị che bởi vật lạ (chăn/gối/khăn/giấy/đồ chơi) — một yếu tố nguy cơ khi ngủ. Theo WHO, SIDS (Hội chứng đột tử ở trẻ sơ sinh) là nguyên nhân tử vong hàng đầu ở trẻ <1 tuổi, mà nhiều case có liên quan đến chăn/gối che mặt khi ngủ. Đề tài em tạo ra hệ thống cảnh báo sớm đúng tình huống này — khi mũi/miệng trẻ bị che, hoặc khi không còn thấy ai trong khung. Em KHÔNG đo hô hấp/nhịp thở (việc đó cần cảm biến y tế chuyên dụng, ngoài phạm vi đồ án); em chỉ phát hiện dấu hiệu quan sát được bằng camera."*

**Q2. Sản phẩm của em phục vụ ai?**

A: *"Cha mẹ trẻ sơ sinh đang nuôi con dưới 1 tuổi. Đặc biệt hữu ích khi cha mẹ phải làm việc, không thể trông trẻ 24/7."*

**Q3. Tại sao em chọn đề tài này?**

A: *"Em thấy thị trường có nhiều baby monitor truyền thống chỉ phát âm thanh và video, không có AI cảnh báo nguy hiểm. Sản phẩm AI thì đắt (Owlet ~5 triệu/cái, cần subscription cloud). Em muốn làm 1 phiên bản tự host trên thiết bị nhúng giá rẻ (Raspberry Pi 4), không cần internet liên tục, dữ liệu không lên cloud — bảo mật riêng tư."*

**Q4. Đề tài em có gì khác biệt so với sản phẩm thương mại?**

A: *"3 điểm khác biệt: (1) Chạy 100% local trên Raspberry Pi 4, không cần cloud → không lo lộ video con. (2) Logic phát hiện đơn giản, dựa vào tỷ lệ màu da nên rất ít báo nhầm trên mặt nhìn rõ. (3) Mã nguồn mở, cha mẹ kỹ thuật có thể tự deploy, tự sửa threshold."*

**Q5. Đâu là điểm khó nhất khi làm đồ án?**

A: *"Khó nhất là TRÁNH BÁO NHẦM trên mặt nhìn rõ. Logic cũ của em từng dùng nhiều tín hiệu đo độ nét ảnh — mặt áp sát camera hay hơi mờ thì các tín hiệu đó tụt và bị hiểu nhầm là 'bị che' dù mặt vẫn sạch. Em đã viết lại, rút gọn về CHỈ dùng tỷ lệ màu da: mặt sạch luôn nhiều da nên không bao giờ kích, còn vật che (chăn/gối/khăn/giấy) thì mất màu da → skin tụt mạnh → báo đúng. Đây là bài học thực nghiệm quan trọng nhất của em."*

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

A: *"Em dùng MediaPipe để định vị tọa độ pixel của mũi và miệng trong khung hình. Sau đó cắt 1 patch 70x70 xung quanh, tính tỷ lệ màu da (skin ratio) trong patch đó. Nếu skin ratio tụt sâu so với baseline (mất màu da → có vật che) ở vùng mũi HOẶC miệng → coi là bị che. Em cũng tính histogram màu nhưng chỉ để hiển thị/log, không dùng để quyết định."*

**Q15. <u>Em dùng những tín hiệu nào và vì sao chỉ skin quyết định?</u>**

A: *"Em tính 2 tín hiệu, nhưng chỉ 1 cái quyết định:*
- *Skin ratio (QUYẾT ĐỊNH) — đếm tỷ lệ pixel có HSV nằm trong khoảng tone da. Mặt sạch luôn nhiều da; chăn/gối/khăn/giấy che → da biến mất → skin tụt → báo bị che.*
- *Histogram HSV correlation (chỉ hiển thị/log) — đo độ giống phân bố màu so với baseline. Em KHÔNG dùng để quyết định vì khi bé cử động môi, histogram vùng miệng tụt oan dù mặt vẫn sạch.*
*Em đã bỏ hẳn edge density và Laplacian variance vì chúng đo độ nét ảnh chứ không đo có vật che — gây báo nhầm trên mặt nhìn rõ nhưng hơi mờ/áp sát camera."*

**Q16. Histogram HSV là gì? Em tính như thế nào?**

A: *"Histogram là biểu đồ phân bố tần suất. Em chuyển ảnh BGR sang HSV, sau đó tính histogram 2D trên H (hue) và S (saturation), bỏ V (value) để bớt nhạy với độ sáng. Sử dụng cv2.calcHist với 36 bin cho H và 32 bin cho S, normalize về [0,1]. Khi check, em so sánh histogram hiện tại với baseline bằng cv2.compareHist với method CORREL — trả về [-1, 1]. Càng gần 1 càng giống. Lưu ý: giá trị này chỉ để hiển thị/log, không tham gia quyết định bị che."*

**Q17. Tại sao bỏ kênh V (Value)?**

A: *"V phản ánh độ sáng. Em không muốn nhạy với độ sáng vì đèn trong phòng có thể đổi (mây che, cha mẹ bật/tắt đèn). Chỉ dùng H (màu sắc) và S (độ bão hòa) → invariant tương đối với lighting."*

**Q18. Skin ratio em tính như thế nào?**

A: *"Em chuyển patch sang HSV rồi dùng cv2.inRange với 2 dải tone da (vì Hue của màu da wrap quanh 0/180). Đếm số pixel nằm trong dải da chia tổng số pixel = skin ratio. Mặt sạch ~0.8-1.0; chăn/gối/khăn/giấy thì gần 0. Đây là tín hiệu bền nhất với mặt sạch (không phụ thuộc độ nét ảnh)."*

**Q19. Tại sao bỏ Laplacian variance và edge density?**

A: *"Vì chúng đo ĐỘ NÉT/CHI TIẾT của ảnh chứ không đo có vật che. Trên mặt nhìn rõ nhưng hơi mờ hoặc áp sát camera, edge và lap_var tụt về gần 0 → bị hiểu nhầm là 'bị che' dù skin=1.0, hist=0.96 (mặt vẫn sạch). Em đã gặp đúng các event báo nhầm này nên loại bỏ 2 tín hiệu đó, chỉ giữ skin ratio."*

**Q20. <u>Hiển thị vote dạng /2 nghĩa là gì? Vote nào thực sự quyết định?</u>**

A: *"Mỗi vùng (mũi/miệng) hiển thị vote dạng `/2` gồm phiếu skin và phiếu hist cho người xem dễ theo dõi. NHƯNG quyết định bị che CHỈ dựa vào phiếu skin: nếu skin tụt dưới ngưỡng ở mũi HOẶC miệng → bị che. Phiếu hist chỉ để hiển thị/log, không làm thay đổi quyết định."*

**Q21. Threshold của em hardcode hay adaptive?**

A: *"Adaptive. Ngưỡng skin tính từ baseline calibration của user: `skin_min = mean_skin × 0.5` (báo khi skin tụt hơn 50%), kèm sàn tuyệt đối 0.15 để tránh ngưỡng quá thấp. Ngưỡng histogram cũng được tính nhưng kẹp bảo thủ trong [0.30, 0.55] và chỉ dùng hiển thị."*

**Q22. Calibration là gì? Tại sao cần?**

A: *"Calibration là pha 5 giây đầu khi hệ thống thấy mặt trẻ. Trong 5 giây này em ghi mean skin ratio (và histogram) của mặt sạch, gọi là baseline. Ngưỡng skin tính từ baseline. Cần vì mỗi đứa trẻ có tone da khác, ánh sáng phòng khác — threshold cố định sẽ false positive hoặc false negative."*

**Q23. Nếu calibration kém thì sao?**

A: *"Em có cơ chế gate: nếu mean_skin < 15% → tức là MediaPipe trỏ landmark vào tóc/quần áo chứ không phải da → reject calibration, yêu cầu calibrate lại. Hoặc nếu standard deviation của histogram correlation > 0.25 → quá hỗn loạn (có thể trẻ cử động nhiều) → cũng reject. Người dùng có thể nhấn phím R để recalibrate thủ công."*

**Q24. <u>Baseline có cập nhật theo thời gian không?</u>**

A: *"Có, em có cơ chế adaptive baseline update. Khi đang ở SAFE và corr > 0.92 (rất ổn), em cập nhật baseline (cả skin và histogram) với learning rate 0.005 (rất chậm) bằng moving average. Mục đích: thích nghi với drift môi trường (ánh sáng đổi dần khi chiều xuống). NHƯNG em KHÔNG update khi đang trong sự kiện che hoặc vừa thoát ra 300 frame (~10s) — tránh học vào trạng thái xấu."*

**Q25. Nếu mặt trẻ thay đổi nhiều (cử động, ngáp...) có bị báo nhầm không?**

A: *"Em xử lý nhiều lớp: (1) Quyết định chỉ dựa vào skin — cử động/ngáp không làm mất màu da nên skin không tụt. (2) Smoothing buffer cần 7/10 frame liên tiếp mới confirm. (3) State machine cần 15 giây liên tục mới báo, kèm anti-flicker 1.5s. Trẻ ngáp 1-2 giây không trigger thông báo."*

### D. Câu hỏi về State Machine (8 câu)

**Q26. Tại sao em dùng state machine?**

A: *"Để biến input liên tục (mỗi frame là 1 boolean 'occluded/not') thành quyết định rời rạc 'alert/safe' có nhớ. State machine theo dõi thời gian liên tục bị che, biết khi nào reset, khi nào fire alert."*

**Q27. Em có mấy state, kể tên?**

A: *"3 state cấp UI: SAFE (có người, mũi/miệng nhìn rõ), COVERED (có người, mũi/miệng bị che — đang đếm/đã báo), NO_PERSON (không thấy ai trong khung). Cộng pha CALIBRATING (5s đầu học baseline) khi mới khởi động."*

**Q28. Khi nào chuyển từ SAFE sang COVERED?**

A: *"Khi có người trong khung VÀ detector báo vùng mũi/miệng mất màu da (skin tụt). Đặt timer `covered_start = now`. Tiếp tục check mỗi frame. Nếu elapsed >= 15s → gửi thông báo 'Mũi/miệng bị che'; nếu còn kéo dài thì nhắc lại định kỳ."*

**Q29. Khi nào chuyển từ COVERED về SAFE?**

A: *"Khi mũi/miệng sạch trở lại. Em có anti-flicker: phải sạch liên tục đủ lâu (`safe_recovery_sec`, mặc định 1.5s) mới reset bộ đếm `covered_start` về None → 1 frame nhiễu không làm reset oan."*

**Q30. <u>Khi không thấy mặt trẻ thì sao? Báo ngay không?</u>**

A: *"Không báo ngay vì 'không thấy mặt' chưa chắc là 'không có người'. 'Có người' = MediaPipe thấy mặt HOẶC YOLO thấy người, và em còn giữ 'có người' thêm `PRESENCE_HOLD_SEC` (mặc định 2s) sau lần thấy gần nhất. Chỉ khi thật sự không còn ai (cả MediaPipe lẫn YOLO đều không thấy) liên tục 15s → mới báo 'Không thấy ai trong khung'."*

**Q31. <u>Làm sao phân biệt 'bé quay đầu/úp mặt (còn người)' vs 'không còn ai'? Cả hai đều không thấy mặt.</u>**

A: *"Em dùng YOLO để xác định CÓ NGƯỜI. Khi MediaPipe không thấy mặt (bé quay đầu/úp mặt) nhưng YOLO vẫn thấy thân người → hệ thống biết VẪN CÒN người → KHÔNG báo 'mất người', đồng thời không tạo cảnh báo che mới vì lúc đó không quan sát được mũi/miệng. Chỉ khi cả MediaPipe lẫn YOLO đều không thấy (quá thời gian giữ presence) → mới báo 'Không thấy ai trong khung'. Đây là một thông báo sự việc, KHÔNG suy diễn nguy hiểm."*

**Q32. Có thể báo mà không cần đếm 15 giây không?**

A: *"Không. Đếm 15s là cơ chế lọc cuối cùng để tránh báo nhầm cho cả 2 loại thông báo. Em có thể tinh chỉnh OCCLUSION_THRESHOLD_SEC (che) và NO_PERSON_SEC (mất người) qua env var nếu thấy cần."*

**Q33. Một thông báo có spam Telegram không nếu tình huống kéo dài?**

A: *"Không spam vô tội vạ. Em có cooldown 60 giây ở tầng gửi (defense-in-depth). State machine quản lý timing chính: sau lần đầu đủ ngưỡng, nếu tình huống còn kéo dài thì NHẮC LẠI định kỳ (mỗi `repeat_sec`, mặc định = ngưỡng) đến khi hết — để cha mẹ không bỏ lỡ. Thông báo đầu của sự kiện mới lưu ảnh; các lần nhắc lại chỉ gửi Telegram, không lưu ảnh trùng."*

### E. Câu hỏi về Telegram (6 câu)

**Q34. Tại sao Telegram mà không SMS?**

A: *"3 lý do: (1) Telegram free, không tốn phí SMS. (2) Telegram gửi được kèm ẢNH — quan trọng để cha mẹ thấy ngay đang bị che cái gì. (3) Telegram Bot API đơn giản, async, không cần SIM. SMS phải qua gateway hoặc GSM module — đắt và phức tạp."*

**Q35. Tin nhắn Telegram chứa gì?**

A: *"Có 2 mẫu tin tương ứng 2 loại thông báo. (1) Khi mũi/miệng bị che: tiêu đề `🚨 MŨI/MIỆNG CỦA BÉ ĐANG BỊ CHE!`, kèm thời gian, số giây bị che, và chi tiết vote mỗi vùng dạng `/2` (hist + skin) cho mũi và miệng. (2) Khi mất người: tiêu đề `⚠️ KHÔNG THẤY AI TRONG KHUNG`, kèm thời gian, số giây đã mất người, nhắc kiểm tra bé/camera. Cả hai đều kèm ảnh JPG quality 92 chụp ngay thời điểm đó. Cha mẹ thấy ngay tình huống thực tế."*

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

A: *"Trên Raspberry Pi 4 CPU-only: ~6-8 FPS end-to-end ở 640×480. Breakdown: MediaPipe 60-90ms, detector skin-ratio 6-12ms, YOLO (chạy mỗi 10 frame, amortized) ~15-25ms, camera read 3ms. Tổng ~120-180ms/frame. State machine chỉ cần ≥6 FPS để hoạt động đúng → 6-8 FPS là vừa đủ an toàn (lý do em hạ xuống 480p thay vì 720p)."*

**Q40. Em test đề tài thế nào? Có bao nhiêu test case?**

A: *"Em có 52 unit test, chia 5 file:*
- *test_state_machine.py — 11 test cho FSM: safe flow, báo che fire ở ngưỡng, anti-flicker khi hết che, nhắc lại định kỳ, báo mất người ở ngưỡng, người quay lại xóa NO_PERSON, chỉ báo che khi có người...*
- *test_occlusion_detector.py — 12 test cho detector: calibration, phát hiện chăn (skin thấp), chăn đỏ, mặt sạch có jitter KHÔNG báo nhầm, không update baseline khi đang che...*
- *test_scene_monitor.py — 5 test cho blur-gate + frozen-frame + luma (watchdog).*
- *test_alert_policy.py — 13 test cho logic timing: watchdog cảnh báo/khôi phục, heartbeat, nhắc hiệu chỉnh, cảnh báo khi điều kiện hiệu chỉnh kém (tối/mờ).*
- *test_eval_metrics.py — 11 test cho logic đo lường (precision/recall/FPR/ROC/AUC + chọn ngưỡng tối ưu) dùng cho bộ công cụ đánh giá detector trên dữ liệu thật.*
*Đặc biệt em có test_clear_face_with_jitter_no_false_alert để đảm bảo mặt sạch (kể cả khi landmark nhảy) KHÔNG bị báo nhầm — đây là failure mode em đặc biệt verify."*

**Q41. Tỷ lệ false positive / false negative bao nhiêu?**

A: *"Em test với các kịch bản thực tế:*
- *Mặt sạch 30 phút → 0 báo nhầm (FP rate ~0%).*
- *Chăn phủ mũi/miệng 20 giây → báo che đúng (TP=100%).*
- *Khăn/giấy phủ mặt 20 giây → báo che đúng (TP=100%).*
- *Khung trống chưa đặt trẻ → KHÔNG báo che (vì không có người).*
- *Bé quay đầu/úp mặt nhưng YOLO vẫn thấy người → KHÔNG báo 'mất người'.*
- *Bế bé ra khỏi khung > 15s → báo 'Không thấy ai trong khung' (đúng, đây là thông báo sự việc).*
*Đây là test định tính, chưa có dataset chuẩn để báo cáo định lượng — điểm em sẽ improve."*

**Q42. <u>Nếu hội đồng hỏi "tại sao không có dataset chuẩn?"</u>**

A: *"Vì không có public dataset cho bài toán 'mũi/miệng trẻ bị che'. Em không thể tạo dataset thật (vấn đề đạo đức + an toàn trẻ). Em đang lên kế hoạch hợp tác với khoa Nhi để thu thập video giám sát trẻ ngủ làm baseline dataset."*

**Q43. Em có chạy thực tế trên Raspberry Pi 4 chưa?**

A: *"Em đã cài và chạy thực tế trên Raspberry Pi 4. INSTALL_PI4.md ghi rõ mọi bước. Em cũng đã viết script tự động install_pi4.sh xử lý các edge case như torch 2.4+ pull nvidia_cudnn không cần, numpy/opencv mismatch. Một sản phẩm production-ready với systemd autostart."*

**Q44. Em có tài liệu hướng dẫn không?**

A: *"File MD lớn INSTALL_PI4.md hướng dẫn deploy từ A-Z trên Raspberry Pi 4 gồm flash Raspberry Pi OS 64-bit, cài deps, cấu hình, systemd service, troubleshooting. Kèm TEST_RESULTS.md ghi các test case và tài liệu bảo vệ này."*

### G. Câu hỏi về tương lai (3 câu)

**Q45. Hạn chế của đề tài là gì?**

A: *"Em thẳng thắn: (1) Ánh sáng yếu/ban đêm chưa test kỹ — MediaPipe có thể mất tracking. (2) Khi trẻ úp mặt xuống nệm, MediaPipe không thấy mũi/miệng nên hệ thống KHÔNG tạo được cảnh báo che (chỉ đánh giá được khi thấy mặt); nếu YOLO vẫn thấy thân người thì hệ thống coi là 'còn người' và im lặng — đây là hạn chế em ghi nhận. (3) Đề tài chưa có IR camera cho ban đêm — em đang thiết kế phiên bản v6 thêm IR cam."*

**Q46. Hướng phát triển tiếp theo là gì?**

A: *"3 hướng cụ thể:*
- *Gắn Google Coral USB TPU + convert YOLO sang TFLite Edge TPU để tăng tốc person detection trên Pi 4 (Pi 4 không có NPU sẵn).*
- *Thêm IR camera để giám sát ban đêm.*
- *Tích hợp relay 4 kênh (em đã có sẵn module): kích còi báo động, đèn cảnh báo khi alert. INSTALL_PI4.md §12 đã có thiết kế nháp."*

**Q47. Có thể thương mại hóa không?**

A: *"Có tiềm năng. Chi phí phần cứng (~2.5 triệu: Raspberry Pi 4 + camera + relay + case + nguồn) rẻ hơn nhiều sản phẩm thương mại như Owlet. Mô hình: bán phần cứng + free app, không bắt subscription cloud. Nhưng em cần làm thêm: chứng nhận y tế, hardening case, dev mobile app cho cha mẹ."*

### H. Câu hỏi technical sâu (3 câu)

**Q48. Em có nghĩ đến dùng deep learning end-to-end không?**

A: *"Có nghĩ đến. Có thể train CNN binary classifier nhận đầu vào 1 patch khuôn mặt, output 'occluded/clear'. Nhưng em không chọn vì: (1) Cần dataset hàng nghìn mẫu mặt trẻ sơ sinh — em không có. (2) Black box, khó debug khi sai. (3) Cách của em — dựa vào skin ratio — interpretable: mỗi thông báo em log rõ skin/hist của mũi và miệng, hội đồng có thể truy ngược root cause. Đây là design choice có chủ ý."*

**Q49. Em có thể giải thích cụ thể tại sao skin ratio phân biệt được mặt sạch và vật che?**

A: *"Skin ratio đếm tỷ lệ pixel có HSV nằm trong dải tone da. Mặt sạch — dù sáng, mờ hay áp sát camera — vùng mũi/miệng vẫn là da nên skin ratio cao (~0.8-1.0). Vật che (chăn/gối/khăn/giấy/đồ chơi) có màu khác da nên pixel rơi ra ngoài dải tone da → skin tụt mạnh, thường về gần 0. Vì tín hiệu này không phụ thuộc độ nét ảnh nên mặt sạch không bao giờ bị báo nhầm — đây là lý do em chọn skin ratio làm tín hiệu quyết định."*

**Q50. <u>Code em chạy CPU mà sao đủ nhanh cho real-time?</u>**

A: *"Vì em tối ưu được pipeline:*
- *MediaPipe được Google compile native ARM, không phải Python pure.*
- *OpenCV operations (cvtColor, inRange, calcHist) đều là C-level.*
- *Em chỉ cắt patch nhỏ 70×70 — phép tính chạy trên patch tốn ít hơn ảnh full.*
- *YOLO chạy mỗi 10 frame, không phải mỗi frame.*
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
[Camera USB] → [Raspberry Pi 4: MediaPipe + Skin-ratio Detector + YOLO + FSM] → [Telegram]
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
3. **Test 1**: *"Bây giờ em phủ một mảnh khăn/giấy lên vùng mũi/miệng — sau 15 giây sẽ có thông báo 'Mũi/miệng bị che'."*
   → Đếm to thành tiếng "1, 2, 3..."
   → Telegram kêu, mở ra cho hội đồng xem
4. **Test 2**: *"Tiếp em rời hẳn khung hình rồi quay lại — sau 15 giây không thấy ai sẽ có thông báo 'Không thấy ai trong khung'; nếu quay lại trước 15s thì KHÔNG có thông báo."*
   → Bước ra khỏi khung ~8-10s (DƯỚI 15s), không có notification → quay lại, về SAFE
   → ⚠️ *Lưu ý khi demo: nếu muốn show thông báo 'mất người', đứng ngoài khung > 15s — đây là thông báo sự việc bình thường, không phải lỗi. Nếu hội đồng hỏi, trả lời theo Q31.*

**3:00-4:00 — Highlight kỹ thuật**

Show slide tín hiệu skin-ratio, giải thích NGẮN:
> *"Điểm em muốn nhấn mạnh là cách hệ thống tránh báo nhầm trên mặt nhìn rõ. Em chỉ dùng tỷ lệ màu da để quyết định: mặt sạch luôn nhiều da nên không bao giờ kích; vật che (chăn/gối/khăn/giấy) mất màu da nên skin tụt mạnh → báo đúng. Em đã bỏ các tín hiệu đo độ nét ảnh vì chúng gây báo nhầm khi mặt hơi mờ/áp sát camera. Histogram em vẫn tính nhưng chỉ để hiển thị. Đây là quyết định thiết kế quan trọng của đồ án."*

**4:00-5:00 — Kết luận**

> *"Tổng kết, đề tài em giải quyết bài toán cảnh báo che mũi/miệng (và mất người) real-time, 100% local trên Raspberry Pi 4 giá rẻ. Logic phát hiện đơn giản, dựa vào tỷ lệ màu da nên rất ít báo nhầm. Có 52 unit test pass, đã chạy production trên thiết bị thực. Hướng phát triển tiếp theo là gắn Coral USB TPU tăng tốc person detection, thêm IR camera cho ban đêm, và tích hợp relay đã có sẵn để kích còi báo động hardware."*
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

> *"Em có 52 unit test pass 100%. Đặc biệt test_clear_face_with_jitter_no_false_alert mô phỏng MediaPipe nhảy landmark ngẫu nhiên trên mặt sạch — verify rằng hệ thống KHÔNG báo nhầm dù landmark di chuyển vài pixel."*

### Điểm 5: Production-ready

> *"Đề tài không chỉ chạy được mà còn deploy được: có systemd service, log rotation 7 ngày, cron cleanup events 30 ngày, signal SIGUSR1 cho recalibrate headless, 3 lớp guard refuse to start khi env sai (numpy ≥2 hoặc opencv ≥4.11)."*

### Điểm 6: Hardware awareness

> *"Em chọn cổng USB 3.0 (cổng xanh) cho camera vì USB 2.0 giới hạn băng thông → tụt FPS — bandwidth bottleneck. Em document rõ trong INSTALL_PI4.md §4.3 'Cắm cổng xanh, không cắm cổng đen'."*

### Điểm 7: 3 lớp logic chất lượng/an toàn bổ sung

> *"Em thêm 3 lớp nâng chất lượng, đều rẻ CPU (~5ms, module `scene_monitor.py`):*
> - *Blur-gate (giám sát độ nét toàn cục): khi cả khung mờ (autofocus hunting / motion blur ở FPS thấp), em dùng tín hiệu này để KHÔNG học baseline vào lúc đó và để cảnh báo điều kiện hiệu chỉnh kém — vì baseline mờ là baseline rác. Quyết định 'bị che' của detector chỉ dựa vào skin ratio nên vốn đã bền với mờ; blur-gate chủ yếu phục vụ chất lượng calibration + watchdog camera.*
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

✅ **Nói thật, đừng nói dối**: *"Em không test trên trẻ sơ sinh thật vì vấn đề đạo đức + an toàn — phải có IRB approval. Em test trên ngôi mặt em và bạn em trong các kịch bản mô phỏng: chăn phủ, gối phủ, khăn/giấy phủ mũi/miệng, rời khung (mất người). Đây là hạn chế em chấp nhận. Hướng phát triển: hợp tác khoa Nhi để có IRB approved data."*

### Câu khó 5: "Nếu trẻ úp mặt xuống nệm hoàn toàn thì sao?"

✅ *"Em trả lời thẳng đây là hạn chế. Khi trẻ úp mặt, MediaPipe không thấy mũi/miệng nên detector KHÔNG đánh giá được vùng mũi/miệng → không tạo cảnh báo che cho tình huống này. Nếu YOLO vẫn thấy thân người thì hệ thống coi là 'còn người' và im lặng (không báo 'mất người'). Đây là giới hạn của cách tiếp cận camera + landmark; hướng khắc phục là thêm IR camera ban đêm và mở rộng tín hiệu phát hiện ở phiên bản sau. Em không phóng đại khả năng của hệ thống ở case này."*

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
