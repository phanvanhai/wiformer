# WiFormer v4 — Mô tả kỹ thuật

Code: [models/wiformer_v4.py](models/wiformer_v4.py). Lập luận thiết kế và kết quả: [WiFormer_Paper.md](WiFormer_Paper.md).

**Tóm tắt:** mỗi đoạn ngắn của mỗi kênh (rx, subcarrier) thành một token gộp hai góc nhìn, dạng sóng (Conv1d) và phổ (FFT). Khối 2 là attention giữa các token RX × subcarrier trong từng đoạn, dùng bias vị trí tương đối. Khối 3 là attention theo thời gian giữa các đoạn, dùng bias tương đối có dấu. Một luồng, một classifier, loss là CE.

---

## 1. Kiến trúc

Shape ghi theo SSHAR-ESP: R = 3 RX, S = 56 sub, T = 1000, fs = 200 Hz, W = 64 (0.32 s), stride 32, K = n_win = 30, W/2 = 32 bin phổ, D = 64, H = 4 head, FFN = 256, C = 8 lớp. Với UT-HAR: R = 3, S = 30, T = 250, fs = 125 Hz, W = 32 (0.256 s), stride 16, K = 14, 16 bin, C = 7.

**Độ dài cửa sổ:** W = lũy thừa của 2 gần nhất với 0.3·fs (ESP: 60 → 64; UT-HAR: 37.5 → 32), stride W/2. Conv1 và Conv2 đều có stride 2, nên đầu ra Conv dài W/4 (ESP 16 bước, UT-HAR 8 bước). Pool `flat4` khi đó chia W/4 thành 4 ô **bằng nhau**.

| Thành phần | Vai trò | Chi tiết |
|---|---|---|
| Tiền xử lý | Chuẩn hóa theo mẫu | `sub`: trừ mean theo (rx, sub), chia std theo (mẫu, rx). `none`: giữ đầu vào của loader. Chỉ dùng anten 0 của mỗi RX |
| Khối 1 | Token = hai góc nhìn của một đoạn | Conv1d ×3 → pool flat4 → Linear; rfft → log1p\|X\| → bỏ DC → Linear; `Linear(2D → D)` trên `[v_time ‖ v_freq]` |
| Khối 2 | Bức tranh không gian của từng cửa sổ | Attention trên R·S token + CLS. Bias: cùng RX → `b_sub[\|Δs\|]`, khác RX → `c_cross`, CLS có bias riêng. Bất biến khi hoán vị RX. 2 lớp |
| Khối 3 | Diễn biến giữa các cửa sổ | Bias có dấu `b[i−j]` (kiểu T5/Swin), CLS có bias riêng. 6 lớp |
| Classifier | | LayerNorm · Dropout · Linear(D → C). Loss = CE |

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
:Chia cửa sổ W = 64 (0.32 s), stride 32
(B, 3, 56, K = 30, 64);
:Khối 1 — hai góc nhìn của từng (rx, sub, window), gộp sớm (Hình 2)
token = Linear(2D → D)([v_time ‖ v_freq])
(B, 3, 30, 56, D); <<#E3F2FD>>
:Khối 2 — attention Rx × Sub trong từng window
bias abs(Δsub) trong cùng RX · scalar cho cặp khác RX · CLS tách vị trí · 2 lớp
h_k (B, 30, D); <<#FFF3E0>>
:Khối 3 — Transformer thời gian
bias có dấu b[i − j] · CLS tách vị trí · 6 lớp
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
:(B, 3, 56, 30, 64)
gộp thành N = B·3·30·56 = B·5040 đoạn
(N, 64);
fork
  :<b>Góc nhìn thời gian</b>
  Conv1d 1→32, k = 8, s = 2 · BN · GELU   (N, 32, 32)
  Conv1d 32→64, k = 4, s = 2 · BN · GELU  (N, 64, 16)
  Conv1d 64→64, k = 3, s = 1 · BN · GELU  (N, 64, 16)
  AdaptiveAvgPool1d(4) · flatten           (N, 256)
  (mỗi kênh: 16 bước → 4 ô × 4 bước, giữ thứ tự trong window)
  Linear(256 → D)
  v_time (N, D);
fork again
  :<b>Góc nhìn phổ</b>
  rfft(64) → 33 bin → log1p(abs)
  bỏ DC, giữ 32 bin                        (N, 32)
  Linear(32 → D)
  v_freq (N, D);
end fork
:fuse: Linear(2D → D) trên [v_time ‖ v_freq]
(N, D);
:reshape
(B, 3, 30, 56, D);
stop
@enduml
```

Ghi chú:
- **Bỏ bin DC:** mức tĩnh của đoạn chỉ nằm ở bin DC, nên góc nhìn phổ chỉ mang thông tin dao động. Với `norm=none` (UT-HAR), mức tĩnh vẫn đi vào mô hình qua góc nhìn thời gian.
- **Độ phân giải phổ:** fs/W = 3.125 Hz trên ESP, khoảng 3.9 Hz trên UT-HAR. Cửa sổ chữ nhật.

## 2. Tham số và MACs

Đo bằng `thop` (một mẫu, inference). `thop` không đếm FFT và phép nhân ma trận bên trong attention, chỉ đếm các lớp Linear/Conv.

| Biến thể | ESP params | ESP MACs | UT-HAR params | UT-HAR MACs |
|---|---|---|---|---|
| `full` | 449,168 | 2.408 G | 447,847 | 0.384 G |
| `wo_fft` | 442,960 | 2.377 G | 442,663 | 0.377 G |
| `wo_conv` | 407,408 | 0.544 G | 406,087 | 0.137 G |
| `wo_block1` | 405,296 | 0.533 G | 402,951 | 0.134 G |
| `wo_block2` | 348,896 | 1.905 G | 347,679 | 0.257 G |
| `wo_block3` | 148,952 | 2.399 G | 147,759 | 0.379 G |
| `freq_mean` | 307,136 | 0.040 G | | |
| `hybrid`, `hybrid_drop` | 453,328 | 2.378 G | | |

Góc nhìn phổ chỉ tốn khoảng 6K tham số và 31M MACs. Gần như toàn bộ MACs nằm ở Conv của Khối 1.

## 3. Phân tích sau khi train

`model.analyze` ghi vào thư mục run:

| File | Nội dung |
|---|---|
| `view_contribution.json` | Chuẩn của trọng số `fuse` cho từng nửa (time / freq); test acc và recall theo lớp khi **zero một góc nhìn lúc test** (knock-out): cả hai, chỉ time, chỉ freq |
| `block2_bias.json` | `b_sub[h][\|Δs\|]`, `c_cross[h]`, bias CLS |
| `block3_bias.json` | `b[h][i − j]` theo offset, bias CLS |
| `rx_permutation.json` | Tỉ lệ dự đoán giữ nguyên khi hoán vị vòng các RX (phải bằng 1) |

Knock-out không giống train riêng một góc nhìn: mô hình chưa từng thấy token thiếu một nửa. Nó chỉ cho biết **mô hình đã dựa vào góc nhìn nào**, theo từng lớp. Câu hỏi "mỗi góc nhìn tự nó đạt bao nhiêu" do ablation `wo_fft` và `wo_conv` trả lời.

## 4. Ablation

| Biến thể | Ý nghĩa |
|---|---|
| `full` | v4 đầy đủ |
| `wo_fft` | Token chỉ có Conv, `Linear(D → D)` |
| `wo_conv` | Token chỉ có phổ, `Linear(D → D)` |
| `wo_block1` | `Linear(W → D)` trên đoạn thô |
| `wo_block2` | Khối 2 = trung bình R·S token của đoạn |
| `wo_block3` | Khối 3 = trung bình qua các cửa sổ |
| `freq_mean` | `wo_conv` + `wo_block2`: chỉ phổ, trung bình R·S token (= phổ trung bình qua rx, sub). So với `wo_conv` |
| `hybrid` | Mỗi window hai nhánh: token chỉ Conv → Khối 2 → `h_k`; phổ `log1p\|rfft\|` trung bình qua (rx, sub) → `Linear(W/2 → D)` → `a_k`. `Linear(2D → D)([h_k ‖ a_k])` → Khối 3. Knock-out trong `view_contribution.json` tính trên lớp gộp này (time = `h_k`, freq = `a_k`) |
| `hybrid_drop` | `hybrid` + `--mod_drop=0.5`: khi train, zero toàn bộ `h_k` của một mẫu với xác suất 0.5 |
| `freq_mean_e200` | `freq_mean`, 200 epoch (EXTRA không được đặt `--epochs`) |
| `nonorm` | `full` với `--norm=none`. Dùng cho UT-HAR (giữ mức tĩnh từng kênh) |

Biến thể `abs_pos` trong paper (mã hóa vị trí tuyệt đối/ALiBi) được chạy bằng `--model=wiformer` (v2, cùng chuẩn hóa và công thức train với v4).

## 5. Huấn luyện và dữ liệu

- Adam, lr đỉnh 1e-3, batch 32, warmup 5 epoch rồi cosine về 0 ở epoch 100, **không dừng sớm** (`patience = 0`), checkpoint theo val acc cao nhất, seed 42.
- Split 6:2:2 cho SSHAR (`repeat`, `user`, `position`); UT-HAR dùng split có sẵn.
- SSHAR với `norm=sub`: loader bỏ z-score toàn cục (`normalize=False`), vì chuẩn hóa trong mô hình đã thay thế.
- `--class_weights`: CE có trọng số (danh sách K số hoặc `auto`).

Chạy:
```
python ablation.py --model=wiformer_v4 --data_root=<...> --dataset=sshar_esp --variants full
python ablation.py --model=wiformer_v4 --data_root=<...> --dataset=ut_har --variants nonorm
python ablation.py --model=wiformer_v4 --list
```
Kết quả nằm ở `results/ablation_v4/<dataset>_<split>_<variant>/`. Hướng dẫn Kaggle: [kaggle_cells.txt](kaggle_cells.txt) với `MODEL = 'wiformer_v4'`.
