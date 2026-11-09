# CSI-WiFormer — kiến trúc đang được cài đặt trong code

> Mô tả đúng theo code hiện tại ở [models/wiformer.py](models/wiformer.py), [main.py](main.py),
> [datasets.py](datasets.py), [train.py](train.py) và [ablation.py](ablation.py) (nhánh `ablation`).
> Cấu hình mặc định: Khối 1 gồm 3 lớp Conv1d + flat4, Khối 2 có 2 lớp, Khối 3 có 6 lớp.
> Ký hiệu: `B` batch, `R` = N_rx, `A` = N_ant, `S` = N_sub, `T` số mẫu thời gian,
> `W` window, `St` stride, `D` d_model, `H` số head.

---

## 0. Tổng quan luồng tensor

```
input            (B, R·A·S, T)            # hoặc (B, R·A, S, T) / (B, R, A, S, T)
 │ preprocess:  reshape → lấy antenna[0] → z-score theo từng (rx, sub)
 ▼
                 (B, R, S, T)
 │ Block 1: unfold(W, St) → [Conv1d ‖ FFT] → fuse
 ▼
                 (B, R, n_win, S, D)
 │ Block 2: joint (Rx × Sub) attention trong từng window, PE_sub, CLS
 ▼
                 (B, n_win, D)
 │ Block 3: CLS + Temporal Transformer với ALiBi
 ▼
                 (B, D)
 │ Classifier: LayerNorm → Dropout → Linear
 ▼
logits           (B, num_classes)          # softmax nằm trong CrossEntropyLoss
```

`n_win = (T − W) // St + 1`.

| Dataset | T | W / St | n_win | Độ dài 1 window |
|---|---|---|---|---|
| sshar_esp (200 Hz) | 1000 | 64 / 32 | **30** (chồng 50%) | 320 ms, bước 160 ms |
| ut_har | 250 | 64 / 32 | 6 | |

`--window 0`: không cắt cửa sổ, W = St = T → n_win = 1 (dùng cho ablation).

---

## 1. Cấu hình layout theo dataset

| Dataset    | N_rx | N_ant | N_sub | Ghi chú                                           |
|------------|------|-------|-------|---------------------------------------------------|
| ut_har     | 3    | 1     | 30    | 1 RX × 3 antenna, coi 3 antenna là 3 "view" (90 kênh) |
| sshar_esp  | 3    | 1     | 56    | 168 kênh                                          |
| sshar_asus | 3    | 4     | 56    | chỉ giữ antenna[0]                                |
| xrf55      | 3    | 3     | 30    | 270 kênh, chỉ giữ antenna[0]                      |

Thứ tự kênh trong dataloader là row-major `(rx, antenna, sub)`. Model kiểm tra `số kênh == R·A·S`, nếu không khớp thì báo lỗi.

---

## 2. Chuẩn hoá và preprocess

**Trong dataloader** ([datasets.py](datasets.py), SSHAR): z-score **toàn cục**, `x = (x − mean) / std` với một
mean và một std vô hướng tính trên tập train. Vì là phép biến đổi affine cùng một hệ số cho mọi phần tử, bước
z-score theo mẫu bên dưới triệt tiêu hoàn toàn bước này.

**Trong model** (không có tham số):

1. `reshape → (B, R, A, S, T)`
2. Lấy `antenna[0]` → `(B, R, S, T)` (`--ant_mode all`: mọi cặp (rx, antenna) là một view)
3. Nếu `sample_norm=True` (mặc định): z-score **theo từng mẫu, từng (rx, sub), dọc trục thời gian trên toàn bộ T**
   `x = (x − mean_T) / (std_T + 1e-6)`. Tắt bằng `--no_sample_norm`.

---

## 3. Block 1 — Sub feature extractor (Conv1d ‖ FFT)

Class `SubFeatureExtractor`. Mọi đoạn tín hiệu (rx, window, sub) dài W được xử lý độc lập, **trọng số dùng chung**.

**Unfold (1 lần, dùng chung cho 2 nhánh):**
```
(B, R, S, T) --unfold(W=64, St=32)--> (B, R, S, n_win, W)
             --permute-->             (B, R, n_win, S, W)
             --reshape-->             (N, W),   N = B·R·n_win·S
```

**Nhánh 1 — temporal (`v_time`), mặc định `k1_conv=3`, `k1_pool=flat4`:**
```
(N, 1, 64)
 Conv1d(1→32, k=8, s=2, p=3)  → BN → GELU     → (N, 32, 32)
 Conv1d(32→64, k=4, s=2, p=1) → BN → GELU     → (N, 64, 16)
 Conv1d(64→64, k=3, s=1, p=1) → BN → GELU     → (N, 64, 16)    # lớp thứ 3 (k1_conv ≥ 3)
 AdaptiveAvgPool1d(4)                          → (N, 64, 4)     # flat4
 flatten                                       → (N, 256)
 proj_time: Linear(256 → D)                    → (N, D)
```
- `k1_conv` ∈ {2, 3, 4}: số lớp Conv1d; mỗi lớp thêm là `Conv1d(64→64, k=3)` giữ nguyên độ dài W/4.
- `k1_pool`:
  - `flatN` (mặc định `flat4`): trung bình về N bước rồi nối lại. Với W = 64 (320 ms), mỗi bước ứng với
    khoảng 80 ms. Giữ được **thứ tự thời gian bên trong window** (vd. phân biệt ngồi xuống và đứng lên).
  - `avg`: trung bình toàn cục về 1 bước, mất thứ tự (đây là cấu hình cũ).
- Số kênh lớp đầu `c1 = max(32, feat // 2)`.

**Nhánh 2 — spectral (`v_freq`):**
```
(N, W)
 rfft (tính ở float32)        → (N, W/2+1) complex   # 33 bin
 log1p(|X|)                   → (N, 33)
 bỏ bin 0 (DC)                → (N, W/2)              # 32 bin
 proj_freq: Linear(W/2 → D)   → (N, D)
```
Biên độ FFT không đổi khi đảo chiều thời gian, nên thông tin thứ tự chỉ đến từ nhánh Conv1d.

**Fuse:**
```
cat[v_time, v_freq]           → (N, 2D)
fuse: Linear(2D → D)          → (N, D)   # không có activation / norm sau fuse
reshape                       → (B, R, n_win, S, D)
```

**Ablation** (`--k1_branch`):
- `both` (mặc định): cả 2 nhánh
- `time`: chỉ Conv1d; `freq`: chỉ FFT. Nhánh bị tắt **không được tạo**, `fuse` thành `Linear(D → D)`.
- `none`: bỏ Khối 1, thay bằng `Linear(W → D)` trên window thô.

---

## 4. Block 2 — Joint (Rx × Sub) attention theo từng window

Class `JointRxSubAttention`.

```
(B, R, n_win, S, D) --permute/reshape--> (B·n_win, R·S, D)
```

- **Positional embedding chỉ theo subcarrier (PE_sub)**, không có PE_rx:
  `pe = Linear(D/2 → D)(Embedding(S, D/2)(sub_idx))` → `(S, D)`, lặp lại cho R receiver → `(R·S, D)`, cộng vào token.
  → Cùng 1 subcarrier ở các Rx khác nhau có cùng PE; model phân biệt Rx chỉ qua nội dung.
- Thêm **CLS token** (learnable) ở đầu → `(B·n_win, 1 + R·S, D)`
- `k2_layers` lớp Transformer encoder (mặc định **2**), `H = 4`, FFN = 4D = 256, dropout 0.1, GELU, **pre-LN** (`norm_first=True`)
- Attention giữa **tất cả** R·S token (không mask)
- Lấy output CLS → `(B, n_win, D)`

Mỗi window có 1 + 3·56 = **169 token** (sshar_esp), 1 + 3·30 = 91 token (ut_har).

**Ablation:** `--k2_mode per_rx` (attention riêng từng Rx rồi lấy trung bình), `--k2_mode pool` (bỏ Khối 2,
lấy trung bình R·S token); `--k2_pe rx_sub | none`.

---

## 5. Block 3 — Temporal Transformer với ALiBi

Class `TemporalTransformer`.

- Thêm **CLS token** → `(B, 1 + n_win, D)`
- **Không có positional embedding**; thay bằng **ALiBi** bias cộng vào attention score:
  `bias[h, i, j] = −m_h · |i − j|`, `m_h = 2^(−8h/H)`, h = 1..H
  - Bias tính trên toàn bộ L = 1 + n_win token, **kể cả CLS** (CLS được coi là vị trí 0)
  - Đối xứng (bidirectional), không causal
- `k3_layers` lớp encoder (mặc định **6**), `H = 4`, FFN = 4D = 256, dropout 0.1, GELU, pre-LN
- Lấy output CLS → `(B, D)`

**Ablation:** `--k3_layers 0` (bỏ Khối 3, lấy trung bình các window token); `--k3_pos learned | none`.

---

## 6. Classifier

```
LayerNorm(D) → Dropout(0.1) → Linear(D → num_classes)
```

---

## 7. Chi tiết cài đặt

- **`SafeTransformerEncoderLayer`** (dùng cho Block 2 và 3): kế thừa `nn.TransformerEncoderLayer` nhưng gọi thẳng `F.multi_head_attention_forward`, **bỏ qua fast path** của PyTorch ở eval mode. Lý do: với mask float theo từng head (ALiBi, shape `(B·H, L, L)`), fast path trả cùng một output cho mọi sample → accuracy test = 1/num_classes. `state_dict` giữ nguyên key như lớp gốc.
- **Khởi tạo trọng số:**
  - Linear: Xavier uniform, bias = 0
  - Conv1d: Kaiming normal (nonlinearity="relu"), bias = 0
  - Embedding: trunc normal std = 0.02
  - MHA `in_proj_weight` (Q, K, V): Xavier uniform, bias = 0
  - 2 CLS token (và `pos_embed` nếu `k3_pos=learned`): trunc normal std = 0.02
- `sub_group` (gộp subcarrier kề nhau trước Khối 1) vẫn còn trong code nhưng **không dùng** (mặc định 1).

---

## 8. Siêu tham số mặc định

| Tham số        | Giá trị | Tham số          | Giá trị |
|----------------|---------|------------------|---------|
| window W       | 64      | k2_layers        | 2       |
| stride St      | 32      | k2_heads / k2_ff | 4 / 256 |
| d_model D      | 64      | k3_layers        | **6**   |
| feat (Conv out)| 64      | k3_heads / k3_ff | 4 / 256 |
| k1_conv        | **3**   | k1_pool          | **flat4** |
| k1_branch      | both    | sample_norm      | True    |
| dropout        | 0.1     | activation       | GELU    |
| norm_first     | True    | ant_mode         | first   |

**Training mặc định cho wiformer** (`main.py`, `train.py`):
- Adam, lr = 1e-3, weight_decay = 0, batch = 32, tối đa 100 epoch
- Warmup tuyến tính 5 epoch rồi CosineAnnealingLR (`T_max = epochs − warmup`)
- Chọn checkpoint theo **val accuracy**; dừng sớm khi val acc không cải thiện sau `--patience` epoch (mặc định 10; 0 = tắt)
- Đánh giá test bằng checkpoint tốt nhất trên val
- `--accum_steps k`: tích luỹ gradient (vd. batch 8 × 4 tương đương batch 32 khi thiếu bộ nhớ GPU)

**Chia dữ liệu SSHAR-ESP** (theo [SSHAR_ESP_protocol.md](SSHAR_ESP_protocol.md), cả hai đều ra train 4480 / val 640 / test 1280):
- `--split_mode user --case_split k` (mục 3A của protocol, cross-user): 7 người train, 1 người val, 2 người test
- `--split_mode repeat` (mục 3B, theo lần lặp): dir00 lần 1–7 train, lần 8 val, lần 9–10 test; dir01/02 lần 5 test,
  lần 4 của một hướng làm val, còn lại train

---

## 9. Số tham số và FLOPs (cấu hình mặc định)

| Thành phần                          | sshar_esp (8 lớp) | ut_har (7 lớp) |
|-------------------------------------|-------------|-------------|
| Block 1 — Conv1d + BN (3 lớp)       | 21,216      | 21,216      |
| Block 1 — proj_time (256 → 64)      | 16,448      | 16,448      |
| Block 1 — proj_freq                 | 2,112       | 2,112       |
| Block 1 — fuse                      | 8,256       | 8,256       |
| **Block 1 tổng**                    | **48,032**  | **48,032**  |
| Block 2 — PE_sub (Embedding + Linear) | 3,904     | 3,072       |
| Block 2 — CLS                       | 64          | 64          |
| Block 2 — 2 × encoder layer (49,984/lớp) | 99,968 | 99,968      |
| **Block 2 tổng**                    | **103,936** | **103,104** |
| Block 3 — CLS + 6 × encoder layer   | **299,968** | **299,968** |
| Classifier                          | 648         | 583         |
| **Toàn mô hình**                    | **452,584** | **451,687** |
| FLOPs forward / mẫu                 | 4.67 G      | 0.50 G      |

FLOPs đo bằng `torch.utils.flop_counter.FlopCounterMode`, chỉ tính lượt forward cho 1 mẫu.

Cấu hình cũ (2 Conv1d + avg, Khối 3 4 lớp) có 327,848 tham số trên sshar_esp. Muốn chạy lại cấu hình này, dùng
biến thể `old_full` hoặc truyền `--k1_conv 2 --k1_pool avg --k3_layers 4`.

---

## 10. Ablation ([ablation.py](ablation.py))

Mỗi biến thể chỉ đổi một yếu tố so với `full` (mô hình đầy đủ). Kết quả lưu ở `<result_root>/<dataset>_wiformer_<variant>`.

| Biến thể | Thay đổi | Tham số (sshar_esp) | FLOPs so với full |
|---|---|---|---|
| `full` | mô hình đầy đủ | 452,584 | 1.0x |
| `wo_fft` | chỉ nhánh Conv1d | 446,376 | |
| `wo_conv` | chỉ nhánh FFT | 410,824 | |
| `wo_block1` | `Linear(W → D)` trên window thô | 408,712 | |
| `wo_block2` | trung bình R·S token | 352,616 | |
| `wo_block3` | trung bình các window token | 152,680 | |
| `b_nowin` | không cắt cửa sổ (W = T, 1 window) | 482,536 | 0.39x |
| `b_nooverlap` | cửa sổ không chồng (W = St = 64, 15 window) | 452,584 | 0.50x |
| `no_norm` | tắt z-score theo mẫu | 452,584 | 1.0x |

- Bộ chạy sẵn: `--set core` (full + bỏ từng khối), `--set window` (full + 2 biến thể cửa sổ), `--set design`
  (lựa chọn bên trong khối, PE, chuẩn hoá, kích thước window).
- `--set scale`: các biến thể tăng kích thước khối. Các biến thể này **giữ nguyên nền cấu hình cũ** (`old_full`) để
  kết quả đã chạy trước khi đổi mặc định vẫn giữ đúng ý nghĩa.
- Liệt kê toàn bộ: `python ablation.py --list`.
