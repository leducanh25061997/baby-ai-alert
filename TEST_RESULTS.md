# Test Results — Baby AI Alert

Báo cáo các test case đã viết và **đều PASS** tính đến lần chạy gần nhất.

| Module | Số test | Pass | Fail |
|---|:-:|:-:|:-:|
| `tests/test_state_machine.py` | 11 | ✅ 11 | 0 |
| `tests/test_occlusion_detector.py` | 14 | ✅ 14 | 0 |
| **Tổng** | **25** | **25** | **0** |

Lệnh chạy lại:
```bash
PYTHONIOENCODING=utf-8 python tests/test_state_machine.py
PYTHONIOENCODING=utf-8 python tests/test_occlusion_detector.py
```

---

## A. State Machine — [tests/test_state_machine.py](tests/test_state_machine.py)

Test FSM (`OcclusionStateMachine`) phụ trách quyết định `SAFE / ALERT / NO_FACE / CALIBRATING` và bắn cảnh báo khi đủ ngưỡng 15 giây. Pure logic, không phụ thuộc cv2/mediapipe.

### A.1 `test_safe_flow` ✅
**Mục đích**: Mặt thấy bình thường, không bị che → state = `SAFE`, không alert.
**Coverage**: happy path cơ bản.

### A.2 `test_histogram_alert_fires_at_threshold` ✅
**Mục đích**: Bị che liên tục đúng 15s → bắn alert đúng 1 lần tại t=15.0, không bắn lại ở các frame sau.
**Coverage**: đúng ngưỡng, không spam.

### A.3 `test_histogram_recovery_resets` ✅
**Mục đích**: Bị che 10s rồi hết bị che → reset `occlusion_start`. Lần bị che kế tiếp đếm lại từ đầu (không cộng dồn).
**Coverage**: tránh false alert do tích lũy nhầm.

### A.4 `test_face_lost_critical_bug_fix` ⭐ ✅
**Bug được fix**: trước đây khi mặt bị **phủ kín** (face_mesh không thấy mặt nữa), code reset bộ đếm → KHÔNG BAO GIỜ alert ở kịch bản nguy hiểm nhất.
**Test**: t=0 thấy mặt → t=0.5 mất mặt → mọi frame sau vẫn đếm tiếp → t=15.0 → ALERT.
**Coverage**: kịch bản ngạt thở nghiêm trọng nhất (chăn phủ kín mặt).

### A.5 `test_face_lost_briefly_then_returns_no_alert` ✅
**Mục đích**: Mặt mất 0.5s rồi quay lại sạch → KHÔNG được false alert.
**Coverage**: tolerance với mediapipe glitch ngắn.

### A.6 `test_face_gone_long_no_yolo_resets` ✅
**Mục đích**: Mặt mất quá lâu (> 25s) mà không có YOLO confirm → coi như trẻ rời khung → reset, không alert lặp lại.
**Coverage**: tránh alert spam khi camera trống.

### A.7 `test_yolo_no_person_resets_immediately` ✅
**Mục đích**: YOLO khẳng định không có người (`person_in_frame=False`) → reset NGAY, không alert.
**Coverage**: phân biệt "trẻ rời khung" vs "trẻ bị phủ kín".

### A.8 `test_yolo_person_present_keeps_counting_beyond_grace` ✅
**Mục đích**: Mặt mất hơn 25s nhưng YOLO vẫn thấy người → tiếp tục đếm → alert.
**Coverage**: trẻ vẫn ở đó nhưng face mesh không tracking được (bị phủ + grace timeout).

### A.9 `test_alert_sent_flag_prevents_spam` ✅
**Mục đích**: Sau khi bắn alert ở t=15, suốt 60s tiếp theo bị che liên tục → KHÔNG bắn thêm lần nào.
**Coverage**: chống spam Telegram.

### A.10 `test_no_face_then_alert_then_recovery` ✅
**Mục đích**: Full lifecycle `NO_FACE → SAFE → ALERT → SAFE`.
**Coverage**: transitions giữa các state.

### A.11 `test_grace_period_starts_count_from_face_lost_moment` ✅
**Mục đích**: `occlusion_start` phải đặt = thời điểm mặt mất (không phải lúc step kế tiếp). Đảm bảo alert chính xác 15s từ lúc bị che.
**Coverage**: chính xác về timing.

---

## B. Occlusion Detector — [tests/test_occlusion_detector.py](tests/test_occlusion_detector.py)

Test `OcclusionDetector` — multi-signal voting (4 signal: histogram + skin ratio + edge density + Laplacian variance).

### B.1 `test_compute_signals_basic` ✅
**Mục đích**: `compute_signals(patch)` trả về đủ 4 key (`hist`, `skin_ratio`, `edge_density`, `lap_var`) với giá trị trong khoảng hợp lệ.
**Output kiểm chứng**: face patch synthetic → `skin=1.00 edge=0.011 lap=634`.

### B.2 `test_blanket_has_low_skin_and_low_lap_var` ✅
**Mục đích**: Patch chăn xám đồng nhất → mọi signal đều thấp (skin ≈ 0, edge ≈ 0, lap_var ≈ 0).
**Output**: `skin=0.00 edge=0.000 lap=0`.

### B.3 `test_face_lap_var_higher_than_hand` ⭐ ✅
**Critical discriminator test**: Laplacian variance phải phân biệt rõ mặt và tay.
**Output**: face=1391, hand=3 → **ratio 463.6x**. Đây là core lý do thêm signal `lap_var` để bắt được trường hợp "tay che mặt" mà 3 signal cũ bị bypass.

### B.4 `test_calibration_needs_enough_samples` ✅
**Mục đích**: Calibrate < 30 sample → finalize fail với message rõ ràng. Buộc user calibrate đủ lâu.
**Output**: `"Quá ít sample (mũi=25, miệng=25, cần >=30)"`.

### B.5 `test_calibration_succeeds_on_stable_face` ✅
**Mục đích**: Calibrate với face frame consistent → ready, quality cao.
**Output**: `quality=1.00`.

### B.6 `test_calibration_fails_on_blanket_landmark` ✅
**Quality gate test**: Nếu landmark trỏ vào nơi không có skin (skin% < 15%) → critical fail. Tránh baseline rác.
**Output**: `❌ Calibration mũi kém: skin chỉ 0% — landmark có thể trỏ vào tóc/quần áo...`

### B.7 `test_check_on_same_frame_safe` ✅
**Mục đích**: Calibrate xong → check với frame y hệt → KHÔNG occluded, votes = 0/8 (toàn bộ).
**Output**: `votes=0/8 corr nose=1.00 mouth=1.00`. Không false positive trên mặt sạch.

### B.8 `test_check_on_blanket_alerts` ✅
**Mục đích**: Chăn xám phủ → cả 4 signal đều vote → 4/4 votes trên cả 2 patch → ALERT.
**Output**: `nose=4/4 mouth=4/4`.

### B.9 `test_check_on_red_blanket_still_alerts` ✅
**Edge case**: chăn đỏ — skin HSV range có overlap với red. Vẫn phải detect.
**Output**: `nose=4/4 mouth=4/4` — histogram + edge + lap_var bù lại được, không lừa được multi-signal voting.

### B.10 `test_check_on_hand_alerts` ⭐ ✅
**Bug user báo được fix**: Tay che mặt → skin tone giống mặt → 3 signal cũ (hist + skin + edge) bị lừa, không vote. Phải dùng Laplacian variance để bắt.
**Output**:
- `nose votes=3/4`, `mouth votes=3/4` → ≥2/4 → ALERT
- `face_lap nose=2980 mouth=1701` vs `hand_lap nose=2 mouth=3` → lap_var discriminator rất mạnh

### B.11 `test_hand_stability_under_landmark_drift` ⭐ ✅
**Bug "đôi khi alert đôi khi không"**: mediapipe landmark nhảy khi tay che → patch sampling khác nhau → vote oscillate.
**Test**: 9 vị trí landmark khác nhau (drift ±0.03 normalized ≈ ±10px).
**Output**: `9/9 frame occluded = 100%` — multi-landmark sampling + MIN aggregation + absolute floor đảm bảo ổn định tuyệt đối.

### B.12 `test_reset_clears_state` ✅
**Mục đích**: `reset()` đưa detector về trạng thái fresh — baseline xóa, quality = 0, không ready.
**Coverage**: hỗ trợ recalibrate (hotkey `R` / SIGUSR1).

### B.13 `test_check_returns_none_before_ready` ✅
**Mục đích**: `check()` trước khi calibrate → trả về `None` (không crash, không false alert).
**Coverage**: edge case lúc startup.

### B.14 `test_baseline_does_not_update_in_alert` ✅
**Anti-drift test**: nếu `prev_in_alert=True`, baseline KHÔNG được EMA-update. Tránh baseline "học" trạng thái che thành ra coi là an toàn.
**Coverage**: ngăn baseline drift sang trạng thái nhầm.

---

## Coverage matrix — Mapping bug → test

| Bug đã gặp | Test bảo vệ |
|---|---|
| Mặt bị phủ kín, face_mesh mất tracking → không bao giờ alert | **A.4** `test_face_lost_critical_bug_fix` |
| Tay che mặt → 3 signal cũ bị lừa skin tone → không alert | **B.10** `test_check_on_hand_alerts` |
| "Đôi khi alert đôi khi không" do landmark drift | **B.11** `test_hand_stability_under_landmark_drift` |
| False positive khi calibrate kém (skin% thấp) | **B.6** `test_calibration_fails_on_blanket_landmark` |
| Baseline drift sang trạng thái sai sau alert | **B.14** `test_baseline_does_not_update_in_alert` |
| Spam alert khi che kéo dài | **A.9** `test_alert_sent_flag_prevents_spam` |
| Alert nhầm khi mặt vừa quay đi 1-2 giây | **A.5** `test_face_lost_briefly_then_returns_no_alert` |
| Alert nhầm khi trẻ rời khung lâu | **A.6**, **A.7** |
| Chăn đỏ trùng skin tone bypass detection | **B.9** `test_check_on_red_blanket_still_alerts` |
| Alert bị trễ vì đếm sai timestamp | **A.11** `test_grace_period_starts_count_from_face_lost_moment` |

---

## Coverage matrix — Mapping yêu cầu sản phẩm → test

Yêu cầu gốc: *"Camera liên tục xác nhận sự hiện diện của mũi và miệng trẻ. Nếu các bộ phận này bị che lấp bởi một vật thể lạ như chăn, gối, gấu bông trong khoảng 15-20 giây → cảnh báo Telegram kèm hình ảnh và thời gian."*

| Yêu cầu | Test verify |
|---|---|
| Phát hiện mặt liên tục | A.1 (SAFE flow), A.10 (state transitions) |
| Đếm thời gian che — đúng 15s | A.2 (exact threshold), A.11 (timestamp accuracy) |
| Phát hiện chăn / gối (smooth, non-skin) | B.8 (chăn xám 4/4), B.9 (chăn đỏ 4/4) |
| Phát hiện tay (skin tone giống mặt) | B.3 (lap_var discriminator), B.10 (3/4 votes) |
| Phát hiện che kín hoàn toàn (mediapipe mất tracking) | A.4 (face_lost trigger) |
| Phân biệt "trẻ rời khung" với "trẻ bị phủ" | A.7 (YOLO no-person reset), A.8 (YOLO person keeps counting) |
| Không spam khi đã alert | A.9 |
| Ổn định khi landmark mediapipe jumpy | B.11 (100% stability) |
| Calibration quality gate (refuse nếu baseline xấu) | B.4 (min samples), B.6 (skin% gate) |
| Recalibration on demand | B.12 (reset clears state) |

---

## Cách reproduce 25/25 PASS

```bash
# Trên máy có numpy<2, opencv-python<4.11
cd baby-ai-alert
PYTHONIOENCODING=utf-8 python tests/test_state_machine.py
# 🎉 11/11 test PASS

PYTHONIOENCODING=utf-8 python tests/test_occlusion_detector.py
# 🎉 14/14 test PASS
```

Nếu fail → chạy `bash scripts/fix_env.sh` để fix numpy/opencv version, sau đó chạy lại.

---

## Lưu ý về synthetic frames

Test `test_occlusion_detector` dùng **synthetic frames** (numpy arrays) chứ không phải video thật của trẻ. Lý do:
- Reproducible & deterministic
- CI/CD pipeline chạy được trên server không có camera
- Verify thuật toán độc lập với hardware

Test integration với camera/mediapipe thật → phải chạy thủ công qua `python src/main.py` theo các kịch bản trong [INSTALL.md](INSTALL.md) mục 9.2.
