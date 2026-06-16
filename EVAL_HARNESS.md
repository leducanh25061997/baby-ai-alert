# Bộ công cụ đánh giá (Hướng A) — đo + tinh chỉnh bộ phát hiện che

Mục tiêu: biến *"em nghĩ thuật toán robust"* → **"em ĐO được X% recall ở Y% báo nhầm"**,
và chọn **ngưỡng tối ưu** cho `SKIN_DROP_FRAC` (tín hiệu quyết định "che" hiện tại).

Đây là **công cụ DEV** — chạy trên máy bạn / Pi để chuẩn bị số liệu + bằng chứng cho
báo cáo. KHÔNG phải code chạy production. Logic đo nằm ở
[src/eval_metrics.py](src/eval_metrics.py) (thuần, có 11 unit test riêng).

---

## 1. Quay dữ liệu (bước tốn công nhất — làm kỹ)

**1 session = 1 bối cảnh** (một kiểu sáng / vị trí camera / đối tượng). Mỗi session
có **1 clip hiệu chỉnh** (`calib`) quay mặt/búp bê RÕ để học baseline, rồi các clip
"thuần" một loại để gán nhãn theo tên file:

```
dataset/
  session01_sang/
    calib.mp4          # 5-10s mặt/búp bê RÕ, KHÔNG che  → học baseline
    sach_01.mp4        # không che                        → nhãn 0
    che_tay_01.mp4     # tay che mũi/miệng                → nhãn 1
    che_chan_01.mp4    # chăn phủ                          → nhãn 1
    che_goi_01.mp4     # gối che                           → nhãn 1
  session02_mo/        # cùng cảnh nhưng thiếu sáng / hơi mờ
    calib.mp4
    sach_01.mp4
    che_chan_01.mp4
  session03_toi/
    ...
```

**Quy ước nhãn theo TÊN FILE:** `calib*` = clip hiệu chỉnh (không xuất) ·
`che*` = 1 (bị che) · `sach*` / `clean*` = 0 (không che) · tên khác → bỏ qua.

**Mẹo & lưu ý quan trọng:**
- Vì bài toán là *phát hiện CHE* (không phải nhận diện mặt), dùng **búp bê/mannequin +
  chăn/gối/tay thật** là hợp lệ và rất nên — tránh cần trẻ thật.
- Quay **đa dạng điều kiện**: sáng / mờ / tối, nhiều loại vật che, vài đối tượng.
  Mỗi điều kiện = 1 session riêng (có `calib` riêng).
- Mỗi clip quay **thuần một loại** → gán nhãn cả clip, khỏi gán từng frame.
- **Để dành ≥1 session chỉ để TEST** (không dùng train ở hướng B) → đo trung thực,
  tránh "test trên đúng cảnh đã train".

---

## 2. Trích đặc trưng → `dataset.csv`

```bash
PYTHONIOENCODING=utf-8 python scripts/extract_features.py --dataset dataset --out dataset.csv
```

Script chạy MediaPipe + `OcclusionDetector` thật qua từng clip: hiệu chỉnh trên
`calib` của session, rồi mỗi frame **thấy mặt** xuất 1 dòng gồm tín hiệu thô, phiếu,
quyết định hiện tại, và **đặc trưng tương đối** (dành cho hướng B sau này) + nhãn.
Frame mất mặt được bỏ qua: harness này chỉ đánh giá bộ phát hiện **che mũi/miệng**
(skin/histogram, cần thấy mặt). Việc "có người mà mất mặt" do nhánh `FACE_LOST` ở
state machine xử lý, đánh giá riêng — không thuộc phạm vi harness này.

Cần `cv2` + `mediapipe` (đã có trong requirements). Tuỳ chọn `--calib-sec`, `--mp-conf`.

---

## 3. Đo + chọn ngưỡng → báo cáo + ROC

```bash
PYTHONIOENCODING=utf-8 python scripts/evaluate.py --csv dataset.csv          # bảng + AUC
PYTHONIOENCODING=utf-8 python scripts/evaluate.py --csv dataset.csv --plot   # + roc.png
```

Báo cáo gồm:
- **Rule hiện tại** (skin-drop ở bất kỳ vùng mũi/miệng): precision / recall / **FPR (báo nhầm)** / F1.
- **Quét ngưỡng 0..4**: để thấy đặt ngưỡng nào cho tradeoff nào.
- **AUC** (diện tích dưới ROC): ≥0.9 xuất sắc · ≥0.8 tốt · ≥0.7 khá.
- **3 gợi ý ngưỡng tối ưu**: Youden (cân bằng), F1, và **An toàn** (recall cao nhất
  trong khi báo nhầm ≤ `--max-fpr`, mặc định 0.05 — hợp baby monitor).
- Xuất `roc.csv` (+ `roc.png` nếu có matplotlib).

`--score sum` đổi sang dùng tổng phiếu (0..8) thay vì `max` hai patch (0..4).

---

## 4. Đọc kết quả thế nào

| Chỉ số | Ý nghĩa cho baby monitor |
|---|---|
| **Recall (TPR)** | Bắt được bao nhiêu % ca bị che. Cao = ít BỎ LỌT (quan trọng nhất về an toàn) |
| **FPR** | Báo nhầm trên ca không che. Cao = hay "kêu oan" → người dùng rút điện |
| **Precision** | Trong các lần báo, bao nhiêu % là thật |
| **AUC** | Năng lực phân biệt tổng thể, độc lập ngưỡng |

→ Chọn ngưỡng theo **triết lý safety-first**: ưu tiên recall cao, nhưng ghìm FPR dưới
mức chịu được (dùng gợi ý "An toàn"). Con số chọn được chính là cơ sở để **chỉnh
`SKIN_DROP_FRAC`** trong [src/occlusion_detector.py](src/occlusion_detector.py)
— giờ là *đo được* thay vì *đoán*.

---

## 5. Liên hệ hướng B (sau này, nếu khách hàng cần)

`dataset.csv` đã chứa sẵn các **đặc trưng tương đối** (`*_rel`, `interaction`) →
là đầu vào để train bộ fusion logistic (hướng B). Tức là làm xong hướng A là đã có
luôn dữ liệu cho B. Xem ghi chú thiết kế B trong bộ nhớ dự án.
