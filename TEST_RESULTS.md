# Test Results — Baby AI Alert

Báo cáo các test case đã viết và **đều PASS** tính đến lần chạy gần nhất.

Logic phát hiện đã được **đơn giản hóa**: chỉ còn **2 thông báo**
— **"Che mũi/miệng"** (mũi/miệng bị vật lạ phủ ≥ `OCCLUSION_THRESHOLD_SEC`, mặc định 15s)
và **"Mất người"** (không thấy ai ≥ `NO_PERSON_SEC`, mặc định 15s).
Trạng thái: `SAFE / COVERED / NO_PERSON`. Quyết định "che" **chỉ dựa trên skin-ratio drop**
(mất màu da → có vật che như chăn/gối/khăn); histogram chỉ để log.

| Module | Số test | Pass | Fail |
|---|:-:|:-:|:-:|
| `tests/test_state_machine.py` | 11 | ✅ 11 | 0 |
| `tests/test_occlusion_detector.py` | 12 | ✅ 12 | 0 |
| `tests/test_scene_monitor.py` | 5 | ✅ 5 | 0 |
| `tests/test_alert_policy.py` | 13 | ✅ 13 | 0 |
| `tests/test_eval_metrics.py` | 11 | ✅ 11 | 0 |
| **Tổng** | **52** | **52** | **0** |

Lệnh chạy lại:
```bash
python3 tests/test_state_machine.py      # 🎉 11/11 test PASS
python3 tests/test_occlusion_detector.py # 🎉 12/12 test PASS
python3 tests/test_scene_monitor.py      # 🎉 5/5 test PASS
python3 tests/test_alert_policy.py       # 🎉 13/13 test PASS
python3 tests/test_eval_metrics.py       # 🎉 11/11 test PASS
```
> mediapipe in nhiều dòng WARNING/log ra stderr — bỏ qua, chỉ cần dòng "🎉 N/N test PASS".

---

## Hai lớp lỗi cũ đã được sửa

**(a) "Không thấy mặt = nguy hiểm" → giờ chỉ là "mất người".**
Logic cũ coi việc face_mesh không còn thấy mặt là dấu hiệu bị phủ kín và đẩy thẳng sang
cảnh báo nguy hiểm. Điều đó gây nhầm: trẻ chỉ quay đi, rời khung, hoặc tracking rớt
cũng bị quy là "nguy hiểm". Logic mới tách bạch: **presence** (có người hay không) được
xác định bằng MediaPipe face **HOẶC** YOLO person, giữ `PRESENCE_HOLD_SEC` (2s) để chống
rớt track ngắn. Khi thật sự không thấy ai đủ lâu → chỉ báo **"Mất người"** (`NO_PERSON`),
không còn quy chụp thành tình huống ngạt.

**(b) Tín hiệu kết cấu báo nhầm "che" trên mặt rõ → giờ quyết định bằng skin-drop.**
Logic cũ bỏ phiếu trên nhiều tín hiệu kết cấu (mật độ cạnh + phương sai Laplacian) để
đoán "bị che". Trên mặt rất rõ, các tín hiệu kết cấu này dao động và **báo nhầm bị che**
(false positive — xem chính frame `events/possible_suffocation_risk_20260512_162643.jpg`).
Logic mới **bỏ hoàn toàn** các tín hiệu kết cấu đó; "che" **chỉ** được quyết định khi
**màu da tụt mạnh** so với baseline (skin-ratio drop). Mặt rõ → màu da còn nguyên → không báo.
Vật che thật (chăn/gối/khăn) → mất màu da → báo.

---

## A. State Machine — [tests/test_state_machine.py](tests/test_state_machine.py)

Test FSM (`OcclusionStateMachine`) quyết định `SAFE / COVERED / NO_PERSON` và bắn cảnh báo
khi đủ ngưỡng. Pure logic, không phụ thuộc cv2/mediapipe. Trigger: `covered` / `no_person`.

### A.1 `test_safe_flow` ✅
**Mục đích**: Có người + không che → state = `SAFE`, không báo, `elapsed = 0`.
**Coverage**: happy path cơ bản.

### A.2 `test_covered_alert_fires_at_threshold` ✅
**Mục đích**: Mũi/miệng bị che liên tục đúng 15s → bắn alert đúng 1 lần tại t=15.0
(t=14.9 chưa bắn), và **không bắn lại** ở frame ngay sau (t=15.1) trong cùng chu kỳ.
**Coverage**: đúng ngưỡng, không spam.

### A.3 `test_covered_recovery_anti_flicker` ✅
**Mục đích**: Bị che rồi hết che → chỉ reset `covered_start` **sau khi sạch đủ lâu**
(`safe_recovery_sec` = 1.5s). 0.5s sạch chưa đủ để reset (vẫn `COVERED`); 1.5s sạch → `SAFE`.
**Coverage**: chống nhấp nháy (anti-flicker) khi tín hiệu dao động quanh ngưỡng.

### A.4 `test_covered_repeats_every_threshold` ✅
**Mục đích**: Còn bị che kéo dài → nhắc lại mỗi `threshold_sec`. Trong 75s bắn **4 lần**,
khoảng cách giữa các lần ≈ 15s.
**Coverage**: re-alert định kỳ khi tình huống chưa được xử lý.

### A.5 `test_no_person_fires_at_threshold` ✅
**Mục đích**: Không thấy ai 15s liên tục → state `NO_PERSON`, trigger `no_person`,
bắn alert đúng tại t=15.0.
**Coverage**: ngưỡng "mất người".

### A.6 `test_no_person_repeats` ✅
**Mục đích**: Mất người kéo dài → nhắc lại mỗi chu kỳ; trong 47s nhắc **3 lần** (~15, 30, 45).
**Coverage**: re-alert định kỳ cho "mất người".

### A.7 `test_person_returns_clears_no_person` ✅
**Mục đích**: Đang đếm "mất người", người quay lại → về `SAFE` ngay, dừng đếm (`absent_start = None`).
**Coverage**: phục hồi tức thì khi trẻ trở lại khung.

### A.8 `test_lost_person_clears_covered` ✅
**Mục đích**: Đang đếm "che" mà người biến mất → chuyển sang đếm "mất người", **xóa bộ đếm che**
(`covered_start = None`, `absent_start` được đặt).
**Coverage**: ưu tiên presence; không cộng dồn nhầm "che" khi đã mất người.

### A.9 `test_covered_only_when_person_present` ✅
**Mục đích**: `covered=True` nhưng không có người → vẫn coi là `NO_PERSON` (presence được ưu tiên).
**Coverage**: không thể "bị che" nếu chưa xác nhận có người.

### A.10 `test_no_alert_before_threshold_then_safe` ✅
**Mục đích**: Che ngắn (<15s) rồi hết → không báo; sau khi sạch đủ lâu → về `SAFE`.
**Coverage**: che thoáng qua không gây cảnh báo.

### A.11 `test_backward_compat_properties` ✅
**Mục đích**: Các thuộc tính tương thích ngược `occlusion_start` / `last_alert_at` vẫn dùng được
cho UI đếm ngược: `occlusion_start` = thời điểm bắt đầu che, `last_alert_at` được đặt khi bắn alert.
**Coverage**: UI countdown không vỡ sau khi đổi logic.

---

## B. Occlusion Detector — [tests/test_occlusion_detector.py](tests/test_occlusion_detector.py)

Test `OcclusionDetector` — đã **đơn giản hóa**: mỗi patch (mũi, miệng) chỉ tính
**2 tín hiệu**: `skin_ratio` (tỉ lệ pixel màu da) và `hist` (histogram, chỉ để log/correlation).
Quyết định "che" dựa trên **skin-ratio tụt mạnh** so với baseline calibration.

### B.1 `test_compute_signals_basic` ✅
**Mục đích**: `compute_signals(patch)` chỉ trả về đúng 2 key `{hist, skin_ratio}`;
`hist` shape (36, 32); `skin_ratio` trong [0,1] và > 0.3 trên patch mặt synthetic.
**Output thực**: `skin=1.00`.

### B.2 `test_blanket_has_low_skin` ✅
**Mục đích**: Patch chăn xám đồng nhất → `skin_ratio` < 0.05 (gần như không có pixel da).
**Output thực**: `skin=0.00`.

### B.3 `test_calibration_needs_enough_samples` ✅
**Mục đích**: Calibrate < `MIN_CALIB_SAMPLES` (30) → finalize fail với message rõ ràng, detector chưa ready.
**Output thực**: `"Quá ít sample (mũi=25, miệng=25, cần >=30)"`.

### B.4 `test_calibration_succeeds_on_stable_face` ✅
**Mục đích**: Calibrate với face frame ổn định → ready, quality cao (> 0.8).
**Output thực**: `q=1.00`.

### B.5 `test_calibration_fails_on_blanket_landmark` ✅
**Quality gate**: Nếu landmark trỏ vào nơi không có da (skin% thấp) → critical fail, tránh baseline rác.
**Output thực**: `❌ Calibration mũi kém: skin chỉ 0% — landmark có thể trỏ vào tóc/quần áo...`

### B.6 `test_check_on_same_frame_safe` ✅
**Mục đích**: Calibrate xong → check với frame y hệt → **KHÔNG** occluded, `total_votes = 0/4`,
hist correlation nose/mouth > 0.95. Không false positive trên mặt sạch.
**Output thực**: `votes=0/4`.

### B.7 `test_check_on_blanket_alerts` ✅
**Mục đích**: Chăn xám phủ → màu da biến mất → cả 2 patch đều vote → occluded.
**Output thực**: `nose=2/2 mouth=2/2`.

### B.8 `test_check_on_red_blanket_alerts` ✅
**Edge case**: Chăn đỏ saturate (hue gần skin nhưng S/V khác hẳn) → vẫn không lọt qua skin-mask → occluded.
**Output thực**: `nose=2/2 mouth=2/2`.

### B.9 `test_clear_face_with_jitter_no_false_alert` ⭐ ✅
**REGRESSION cho lớp lỗi cũ (b)** — chính các frame `events/...162626, ...162643`:
mặt sạch có `skin ~1.0` nhưng tín hiệu phụ dao động vẫn bị báo "che". Sau khi bỏ tín hiệu
kết cấu: calibrate trên một mặt, check trên mặt khác (cùng tone da, có nhiễu sáng/màu nhẹ
`skin_jitter=20`) → vẫn còn nhiều màu da → **KHÔNG** occluded.
**Output thực**: `nose hist=0.26 skin=0.96` (hist tụt nhưng skin còn cao → không báo nhầm).

### B.10 `test_reset_clears_state` ✅
**Mục đích**: `reset()` đưa detector về fresh — baseline xóa (`nose = None`, `mouth = None`), không ready.
**Coverage**: hỗ trợ recalibrate (hotkey `R` / SIGUSR1).

### B.11 `test_check_returns_none_before_ready` ✅
**Mục đích**: `check()` trước khi calibrate xong → trả về `None` (không crash, không false alert).
**Coverage**: edge case lúc startup.

### B.12 `test_baseline_does_not_update_in_alert` ✅
**Anti-drift**: nếu `prev_in_alert=True`, baseline (`nose.hist`) **không** được EMA-update.
Tránh baseline "học" trạng thái che thành an toàn.
**Coverage**: ngăn baseline drift sang trạng thái nhầm.

---

## C. Scene Monitor — [tests/test_scene_monitor.py](tests/test_scene_monitor.py)

Test `SceneMonitor` — giám sát điều kiện cảnh để **gate** (không phải để phát hiện che):
độ nét (blur-gate), frozen-frame, độ sáng (luma).

### C.1 `test_blur_gate_detects_global_blur` ✅
**Mục đích**: Học baseline độ nét trên frame nét → frame mờ (sharpness tụt > `blur_drop_frac`) →
`is_blurry=True`; nét trở lại → `False`.

### C.2 `test_blur_gate_off_before_baseline` ✅
**Mục đích**: Chưa có baseline → `is_blurry` luôn `False` (warmup an toàn, không gate nhầm).

### C.3 `test_frozen_frame_detection` ✅
**Mục đích**: Hai frame y hệt (diff ≈ 0) → `is_frozen_frame=True`; cảnh thay đổi → `False`.

### C.4 `test_luma_dark_vs_bright` ✅
**Mục đích**: Đo độ sáng trung bình — frame tối luma < 30, frame sáng luma > 200.

### C.5 `test_reset_clears_state` ✅
**Mục đích**: `reset()` xóa `sharp_baseline` và `_prev_gray` về `None`.

---

## D. Alert Policy — [tests/test_alert_policy.py](tests/test_alert_policy.py)

Test logic timing thuần (tách khỏi main.py, không cần cv2/mediapipe/telegram):
`Watchdog`, `HeartbeatPolicy`, `CalibrationReminder`, `CalibrationConditionWarner`.

### D.1 `test_watchdog_alert_only_after_degrade_sec` ✅
Sự cố < `degrade_sec` (20s) → chưa báo; tại 20s → báo đúng 1 lần (`kind='alert'`); sau đó không lặp.

### D.2 `test_watchdog_recover_after_alert` ✅
Sự cố đã báo rồi hết → phát đúng 1 sự kiện `recover`; tiếp tục hết → không phát lại.

### D.3 `test_watchdog_no_recover_if_never_alerted` ✅
Sự cố thoáng qua (< `degrade_sec`) rồi hết → KHÔNG phát gì cả.

### D.4 `test_heartbeat_disabled` ✅
`interval_sec=0` → không bao giờ fire.

### D.5 `test_heartbeat_fires_on_interval` ✅
Lần đầu chỉ đặt mốc; sau đó fire mỗi `interval_sec` (60s): fire tại 1060, 1120; không fire giữa chừng.

### D.6 `test_calib_reminder_disabled` ✅
`enabled=False` → không bao giờ nhắc.

### D.7 `test_calib_reminder_reminds_when_no_face` ✅
Chưa calib + không thấy mặt → nhắc mỗi `remind_sec` (60s): fire tại 60, 120.

### D.8 `test_calib_reminder_silent_when_face_or_done` ✅
Thấy mặt hoặc đã calib xong → không nhắc + reset mốc; mất mặt lại → đếm lại từ đầu (nhắc tại 131).

### D.9 `test_calib_warner_disabled` ✅
`enabled=False` → không nhắc dù điều kiện kém kéo dài.

### D.10 `test_calib_warner_silent_when_conditions_good` ✅
Điều kiện tốt → không nhắc.

### D.11 `test_calib_warner_silent_when_not_calibrating` ✅
Không trong pha hiệu chỉnh → không nhắc dù điều kiện kém.

### D.12 `test_calib_warner_warns_after_grace_then_repeats` ✅
Điều kiện kém liên tục → nhắc lần đầu sau `grace_sec` (4s), rồi lặp mỗi `remind_sec` (60s): fire tại 4, 64, 124.

### D.13 `test_calib_warner_resets_when_conditions_recover` ✅
Kém chưa quá grace rồi tốt lại → quên; kém lại → phải chờ grace từ đầu (fire tại 9).

---

## E. Eval Metrics — [tests/test_eval_metrics.py](tests/test_eval_metrics.py)

Test logic đo thuần phục vụ bộ công cụ đánh giá: precision/recall/FPR, ROC/AUC, chọn ngưỡng tối ưu —
[src/eval_metrics.py](src/eval_metrics.py). Xem thêm [EVAL_HARNESS.md](EVAL_HARNESS.md).

### E.1 `test_confusion_basic_metrics` ✅
`Confusion(tp,fp,tn,fn)` → precision/recall/fpr/specificity/accuracy/f1 đúng công thức.

### E.2 `test_confusion_zero_division_safe` ✅
Confusion toàn 0 → mọi chỉ số = 0.0, không chia 0.

### E.3 `test_confusion_at_threshold` ✅
`confusion_at(scores, labels, t)` đúng theo rule "đoán dương khi score >= t".

### E.4 `test_roc_perfect_separation_auc_1` ✅
Phân tách hoàn hảo → ROC chứa điểm (fpr=0, tpr=1), AUC = 1.0.

### E.5 `test_roc_no_discrimination_auc_half` ✅
Mọi score bằng nhau → AUC = 0.5.

### E.6 `test_roc_partial_separation_auc` ✅
Phân tách một phần → AUC = 0.75.

### E.7 `test_roc_endpoints_present` ✅
ROC luôn chứa hai điểm mút (0,0) và (1,1).

### E.8 `test_sweep_vote_thresholds` ✅
`sweep` qua nhiều ngưỡng → trả đủ điểm; tại threshold=2 confusion đúng (tp=4, fp=0, tn=4, fn=0).

### E.9 `test_best_recall_at_fpr_safety_first` ✅
Chọn ngưỡng recall cao nhất **trong giới hạn FPR** (≤ 0.05) — an toàn trước, loại điểm vượt trần.

### E.10 `test_best_recall_at_fpr_none_when_all_exceed` ✅
Nếu không điểm nào đạt trần FPR → trả về `None`.

### E.11 `test_best_by_youden_and_f1_run` ✅
`best_by_youden` (J = tpr − fpr) và `best_by_f1` chạy đúng; phân tách hoàn hảo → J=1, F1=1.

---

## Kiểm chứng tích hợp trên ẢNH THẬT (false-positive cũ đã được sửa)

Bằng chứng cho **lớp lỗi (b)**: dùng đúng frame mặt-rõ
[events/possible_suffocation_risk_20260512_162643.jpg](events/possible_suffocation_risk_20260512_162643.jpg)
— frame này TỪNG bị logic cũ báo nhầm "bị che". Quy trình: dùng MediaPipe FaceMesh lấy landmark
trên chính ảnh → calibrate detector trên chính frame đó → check lại.

| Trường hợp | occluded | skin (mũi / miệng) |
|---|:-:|:-:|
| Check trên chính frame mặt rõ | **False** ✓ | 1.00 / 1.00 |
| Phủ chăn xám lên vùng mũi/miệng | **True** ✓ | 0.00 / 0.03 |

Calibration trên frame thật: `ok=True, quality=1.00`. Kết quả khẳng định: mặt rõ **không**
còn bị báo nhầm (skin còn nguyên → SAFE), trong khi vật che thật khiến skin tụt sâu → báo đúng.

---

## Lưu ý về synthetic frames

Phần lớn test `test_occlusion_detector` / `test_scene_monitor` dùng **synthetic frames**
(numpy arrays) thay vì video thật của trẻ, vì:
- Reproducible & deterministic.
- CI/CD chạy được trên server không có camera.
- Verify thuật toán độc lập với hardware.

Kiểm chứng tích hợp với camera/mediapipe thật → chạy thủ công qua `python src/main.py`
theo các kịch bản trong [INSTALL_PI4.md](INSTALL_PI4.md).
