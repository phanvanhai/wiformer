# WiFormer v4 — Lập luận thiết kế

**Một câu:** v4 = Khối 1 của v2 (gộp sớm hai góc nhìn thời gian và phổ ngay tại token) + Khối 2, Khối 3 của v3 (bias vị trí tương đối, CLS tách vị trí). Chỉ còn **một luồng**, một classifier, loss chỉ là CE.

Code: [models/wiformer_v4.py](models/wiformer_v4.py). Khối 2, Khối 3 và phần chuẩn hóa dùng lại đúng code của [models/wiformer_v3.py](models/wiformer_v3.py). Lập luận chi tiết về Khối 2 và Khối 3 nằm ở [WiFormer_v3_Architecture.md](WiFormer_v3_Architecture.md), mục 9–11; tài liệu này chỉ tóm tắt.

v4 được đánh giá như một mô hình mới, có bộ ablation riêng (mục 6). Kết quả v2, v3 chỉ dùng để tham khảo.

---

## 1. Vì sao có v4

Kết quả v3 (gộp muộn theo lớp) trên SSHAR-ESP, split repeat 6:2:2, seed 42:

| | test acc | params | MACs |
|---|---|---|---|
| v3 `full` (hai luồng, gộp muộn) | 96.09 | 554K | 2.40 G |
| v3 `wo_main` (chỉ luồng tốc độ) | 96.17 | 115K | 4.1 M |

| Số mẫu nhầm | v3 `full` | v3 `wo_main` |
|---|---|---|
| ngồi xuống ↔ đứng lên | **5** | 14 |
| đi ↔ chạy | 24 | **10** |
| nằm im ↔ phòng trống | **16** | 20 |

Hai luồng bổ sung cho nhau theo cặp lớp, nhưng gộp muộn không khai thác được điều đó:
- Trọng số theo lớp `w_c` chỉ nằm trong khoảng 0.43–0.59, gần như lấy trung bình logit.
- Trong `full`, luồng tốc độ nhận đúng đi/chạy tốt hơn bản đã gộp (95.0 / 93.8 so với 93.8 / 91.3).
- Khi train chung, luồng tốc độ yếu đi (93.75% so với 96.17% khi train riêng). Đây là hiện tượng *modality imbalance* đã được ghi nhận trong học đa phương thức (Wang et al., *Gradient Blending*, CVPR 2020; Peng et al., *OGM-GE*, CVPR 2022).

## 2. Nguyên lý: hai góc nhìn khớp nhau thì gộp sớm

Gộp muộn hợp với các nguồn **khác loại, không thẳng hàng** (ví dụ ảnh và âm thanh). Thời gian và phổ ở đây là **hai góc nhìn của cùng một đoạn CSI**, khớp nhau đến từng ô (rx, sub, window). Với dữ liệu như vậy, gộp ngay tại token là lựa chọn tự nhiên hơn:

| | v3 (gộp muộn theo lớp) | v4 (gộp sớm theo token) |
|---|---|---|
| Hai nguồn có thẳng hàng không | Mất thẳng hàng: phổ đã lấy trung bình qua sub và gộp qua RX trước khi gộp | Có: cùng đoạn, cùng ô (rx, sub, window) |
| Chọn subcarrier có phổ tốt | Không ai chọn; trung bình làm loãng phổ (v3 mục 18) | Khối 2 attention tự chọn |
| Tương tác thời gian × tần số | Chỉ ở logit cuối | Ở mọi tầng của Khối 2 và Khối 3 |
| Modality imbalance | Có | Không có (một luồng) |
| Diễn giải "lớp nào dựa vào nguồn nào" | Có (`w_c`, recall từng luồng) | Gián tiếp: knock-out lúc test (mục 5) và ablation |

Phổ trong **một** cửa sổ không biết chiều thời gian, nhưng **chuỗi** token qua Khối 3 với bias có dấu thì biết. Vì vậy thông tin nhịp điệu bất đối xứng (ví dụ ngồi xuống dừng gấp, đứng lên tắt dần) vẫn được giữ, giống luồng tốc độ của v3.

## 3. Kiến trúc

Shape ghi theo SSHAR-ESP: R = 3 RX, S = 56 sub, T = 1000, fs = 200 Hz, W = 60 (0.3 s), stride 30, K = n_win = 32, W/2 = 30 bin phổ, D = 64, H = 4 head, FFN = 256, C = 8 lớp. Với UT-HAR: R = 3, S = 30, T = 250, fs = 125 Hz, W = 36, stride 18, K = 12, 18 bin, C = 7.

| Thành phần | Vai trò | Nguồn | Chi tiết |
|---|---|---|---|
| Tiền xử lý | Chuẩn hóa theo mẫu | v3 mục 9 | `sub`: trừ mean theo (rx, sub), chia std theo (mẫu, rx). `none`: giữ đầu vào của loader |
| Khối 1 | Token = hai góc nhìn của một đoạn | v2 | Conv1d ×3 → pool flat4 → Linear; rfft → log1p\|X\| → bỏ DC → Linear; `Linear(2D → D)` trên `[v_time ‖ v_freq]` |
| Khối 2 | Bức tranh không gian của từng cửa sổ | v3 mục 10 | Attention trên R·S token + CLS. Bias: cùng RX → `b_sub[\|Δs\|]`, khác RX → `c_cross`. Bất biến khi hoán vị RX. 2 lớp |
| Khối 3 | Diễn biến giữa các cửa sổ | v3 mục 11 | Bias có dấu `b[i−j]` (T5/Swin), CLS tách vị trí. 6 lớp |
| Classifier | | v3 | LayerNorm · Dropout · Linear(D → C). Loss = CE |

### Hình 1 — Tổng quan

```plantuml
@startuml
skinparam defaultFontName Arial
skinparam ActivityBorderColor #555555
skinparam ArrowColor #555555
title WiFormer v4 — tổng quan

start
:CSI biên độ đầu vào
(B, R·N_ant·S, T = 1000);
:Tiền xử lý
chọn anten[0] · sub: trừ mean theo (rx, sub), chia std theo (mẫu, rx) · none: giữ nguyên
(B, R = 3, S = 56, T = 1000);
:Chia cửa sổ W = 60 (0.3 s), stride 30
(B, 3, 56, K = 32, 60);
:Khối 1 — hai góc nhìn của từng (rx, sub, window), gộp sớm (Hình 2)
token = Linear(2D → D)([v_time ‖ v_freq])
(B, 3, 32, 56, D); <<#E3F2FD>>
:Khối 2 — attention Rx × Sub trong từng window
bias abs(Δsub) trong cùng RX · scalar cho cặp khác RX · CLS tách vị trí · 2 lớp
h_k (B, 32, D); <<#FFF3E0>>
:Khối 3 — Transformer thời gian
bias T5 có dấu b[i − j] · CLS tách vị trí · 6 lớp
(B, D); <<#E8F5E9>>
:Classifier
LayerNorm · Dropout · Linear(D → C)
(B, C)
loss = CE; <<#FCE4EC>>
stop
@enduml
```

### Hình 2 — Khối 1: hai góc nhìn, gộp sớm

```plantuml
@startuml
skinparam defaultFontName Arial
skinparam ActivityBackgroundColor #E3F2FD
skinparam ActivityBorderColor #555555
skinparam ArrowColor #555555
title Khối 1 — trọng số dùng chung cho mọi (rx, window, sub)

start
:(B, 3, 56, 32, 60)
gộp thành N = B·3·32·56 = B·5376 đoạn
(N, 60);
fork
  :<b>Góc nhìn thời gian</b>
  Conv1d 1→32, k = 8, s = 2 · BN · GELU   (N, 32, 30)
  Conv1d 32→64, k = 4, s = 2 · BN · GELU  (N, 64, 15)
  Conv1d 64→64, k = 3, s = 1 · BN · GELU  (N, 64, 15)
  AdaptiveAvgPool1d(4) · flatten           (N, 256)
  Linear(256 → D)
  v_time (N, D);
fork again
  :<b>Góc nhìn phổ</b>
  rfft(60) → 31 bin → log1p(abs)
  bỏ DC → 30 bin                           (N, 30)
  Linear(30 → D)
  v_freq (N, D);
end fork
:fuse: Linear(2D → D) trên [v_time ‖ v_freq]
(N, D);
:reshape
(B, 3, 32, 56, D);
stop
@enduml
```

Ghi chú:
- **Bỏ bin DC:** mức tĩnh của đoạn chỉ nằm ở bin DC, nên góc nhìn phổ chỉ mang thông tin dao động. Với `norm=none` (UT-HAR), mức tĩnh vẫn đi vào mô hình qua góc nhìn thời gian.
- **Khác v2:** v2 dùng W = 64 cố định (32 bin). v4 dùng cửa sổ 0.3 s theo fs như v3, nên số bin phụ thuộc dataset (ESP 30 bin, độ phân giải khoảng 3.3 Hz; UT-HAR 18 bin, khoảng 3.5 Hz).

## 4. Tham số và MACs

Đo bằng `thop` (một mẫu, inference), cùng cách đo với v3. `thop` không đếm FFT và phép nhân ma trận bên trong attention, chỉ đếm các lớp Linear/Conv.

| Biến thể | ESP params | ESP MACs | UT-HAR params | UT-HAR MACs |
|---|---|---|---|---|
| `full` | 449,056 | 2.451 G | 447,959 | 0.353 G |
| `wo_fft` | 442,976 | 2.418 G | 442,647 | 0.347 G |
| `wo_conv` | 407,296 | 0.579 G | 406,199 | 0.118 G |
| `wo_block1` | 405,056 | 0.568 G | 403,191 | 0.115 G |
| `wo_block2` | 348,784 | 1.914 G | 347,791 | 0.244 G |
| `wo_block3` | 148,824 | 2.441 G | 147,887 | 0.349 G |

- So với v3 `full` (554K, 2.40 G), v4 ít hơn khoảng 105K tham số vì bỏ luồng tốc độ và Khối 3'. MACs gần như không đổi.
- Góc nhìn phổ chỉ tốn khoảng 6K tham số và 33M MACs (2 Linear trên 5376 token). Gần như toàn bộ MACs nằm ở Conv của Khối 1, nên `wo_conv` rẻ hơn khoảng 4 lần.

## 5. Phân tích sau khi train

`model.analyze` ghi vào thư mục run:

| File | Nội dung |
|---|---|
| `view_contribution.json` | Chuẩn của trọng số `fuse` cho từng nửa (time / freq); test acc và recall theo lớp khi **zero một góc nhìn lúc test** (knock-out): cả hai, chỉ time, chỉ freq |
| `block2_bias.json` | `b_sub[h][\|Δs\|]`, `c_cross[h]`, bias CLS |
| `block3_bias.json` | `b[h][i − j]` theo offset, bias CLS |
| `rx_permutation.json` | Tỉ lệ dự đoán giữ nguyên khi hoán vị vòng các RX (phải bằng 1) |

Knock-out không giống train riêng một góc nhìn: mô hình chưa từng thấy token thiếu một nửa. Nó chỉ cho biết **mô hình đã dựa vào góc nhìn nào**, theo từng lớp. Câu hỏi "mỗi góc nhìn tự nó đạt bao nhiêu" do ablation `wo_fft` và `wo_conv` trả lời.

## 6. Ablation

| Biến thể | Ý nghĩa | Câu hỏi trả lời |
|---|---|---|
| `full` | v4 đầy đủ | |
| `wo_fft` | Token chỉ có Conv, `Linear(D → D)` | Góc nhìn phổ có đóng góp không. Cùng cấu trúc với v3 `wo_freq` |
| `wo_conv` | Token chỉ có phổ, `Linear(D → D)` | Phổ từng sub qua Khối 2–3 có đủ không, so với luồng tốc độ của v3 (`wo_main`) |
| `wo_block1` | `Linear(W → D)` trên đoạn thô | Khối 1 có cần không |
| `wo_block2` | Trung bình R·S token | Attention Rx × Sub có cần không |
| `wo_block3` | Trung bình qua các cửa sổ | Diễn biến thời gian có cần không |
| `nonorm` | `full` với `--v3_norm=none` | Thí nghiệm riêng cho UT-HAR (giữ mức tĩnh từng kênh, v3 mục 16.3b) |

So sánh chính:
- **v4 `full` với v3 `full`:** gộp sớm hay gộp muộn tốt hơn, cùng split, cùng công thức train.
- **v4 `wo_conv` với v3 `wo_main`:** phổ từng sub để Khối 2 chọn, hay phổ lấy trung bình qua sub.

## 7. Các dự đoán kiểm chứng được

1. **Đi ↔ chạy:** v4 `full` ít lỗi hơn v3 `full` (24) và gần bằng v3 `wo_main` (10), vì phổ không còn bị pha loãng bởi gộp muộn.
2. **Ngồi xuống ↔ đứng lên, nằm im ↔ phòng trống:** v4 `full` giữ được mức của v3 `full` (5 và 16), vì góc nhìn thời gian và Khối 2 vẫn còn.
3. **`wo_conv` ≥ v3 `wo_main`:** Khối 2 chọn được subcarrier nhạy, thay vì lấy trung bình đều.
4. **Knock-out:** zero `v_freq` làm hỏng đi/chạy nhiều nhất; zero `v_time` làm hỏng nằm im/phòng trống và ngồi/đứng nhiều nhất.

Nếu dự đoán 1 sai (v4 `full` vẫn nhiều lỗi đi/chạy), mô hình đang bỏ qua góc nhìn phổ trong `fuse`. Khi đó `view_contribution.json` sẽ cho thấy chuẩn trọng số nửa freq nhỏ, và có thể xem xét các cách ép dùng phổ (ví dụ modality dropout trên từng nửa token).

## 8. Huấn luyện và dữ liệu

Giống v3:
- Adam, lr đỉnh 1e-3, batch 32, warmup 5 epoch rồi cosine về 0 ở epoch 100, **không dừng sớm** (`patience = 0`), checkpoint theo val acc cao nhất, seed 42.
- Split 6:2:2 cho SSHAR (`repeat`, `user`, `position`); UT-HAR dùng split có sẵn.
- SSHAR với `norm=sub`: loader bỏ z-score toàn cục (`normalize=False`), vì chuẩn hóa trong mô hình đã thay thế.
- `--class_weights` dùng được như v3 (CE có trọng số).

Chạy:
```
python ablation.py --model=wiformer_v4 --data_root=<...> --dataset=sshar_esp --variants full
python ablation.py --model=wiformer_v4 --data_root=<...> --dataset=ut_har --variants nonorm
python ablation.py --model=wiformer_v4 --list
```
Kết quả nằm ở `results/ablation_v4/<dataset>_<split>_<variant>/`. Hướng dẫn Kaggle: [kaggle_cells.txt](kaggle_cells.txt) với `MODEL = 'wiformer_v4'`.
