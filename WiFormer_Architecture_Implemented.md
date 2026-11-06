# CSI-WiFormer — kiến trúc đang được cài đặt trong code

> Mô tả đúng theo code hiện tại ở [models/wiformer.py](models/wiformer.py) và [main.py](main.py).
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

`n_win = (T − W) // St + 1`. Ví dụ UT_HAR: T = 250, W = 64, St = 32 → **n_win = 6**.

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

## 2. Preprocess (trong model, không có tham số)

1. `reshape → (B, R, A, S, T)`
2. Lấy `antenna[0]` → `(B, R, S, T)`
3. Nếu `sample_norm=True` (mặc định): z-score **theo từng mẫu, từng (rx, sub), dọc trục thời gian trên toàn bộ T**
   `x = (x − mean_T) / (std_T + 1e-6)`

---

## 3. Block 1 — Sub feature extractor (Conv1d ‖ FFT)

Class `SubFeatureExtractor`. Mọi đoạn tín hiệu (rx, window, sub) dài W được xử lý độc lập, **trọng số dùng chung**.

**Unfold (1 lần, dùng chung cho 2 nhánh):**
```
(B, R, S, T) --unfold(W=64, St=32)--> (B, R, S, n_win, W)
             --permute-->             (B, R, n_win, S, W)
             --reshape-->             (N, W),   N = B·R·n_win·S
```

**Nhánh 1 — temporal (`v_time`):**
```
(N, 1, 64)
 Conv1d(1→32, k=8, s=2, p=3) → BN → GELU     → (N, 32, 32)
 Conv1d(32→64, k=4, s=2, p=1) → BN → GELU    → (N, 64, 16)
 AdaptiveAvgPool1d(1)                         → (N, 64)
 proj_time: Linear(64 → D)                    → (N, D)
```

**Nhánh 2 — spectral (`v_freq`):**
```
(N, W)
 rfft (tính ở float32)        → (N, W/2+1) complex   # 33 bin
 log1p(|X|)                   → (N, 33)
 bỏ bin 0 (DC)                → (N, W/2)              # 32 bin
 proj_freq: Linear(W/2 → D)   → (N, D)
```

**Fuse:**
```
cat[v_time, v_freq]           → (N, 2D)
fuse: Linear(2D → D)          → (N, D)   # không có activation / norm sau fuse
reshape                       → (B, R, n_win, S, D)
```

**Ablation** (`--k1_branch`):
- `both` (mặc định): cả 2 nhánh
- `time`: v_freq = 0
- `freq`: v_time = 0

Nhánh bị tắt vẫn còn tham số trong model (không dùng), vẫn đi qua `fuse`.

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
- `k2_layers` lớp Transformer encoder (mặc định **2**), `H = 4`, FFN = 256, dropout 0.1, GELU, **pre-LN** (`norm_first=True`)
- Attention giữa **tất cả** R·S token (không mask)
- Lấy output CLS → `(B, n_win, D)`

Ví dụ UT_HAR: mỗi window có 1 + 3·30 = **91 token**.

---

## 5. Block 3 — Temporal Transformer với ALiBi

Class `TemporalTransformer`.

- Thêm **CLS token** → `(B, 1 + n_win, D)`
- **Không có positional embedding**; thay bằng **ALiBi** bias cộng vào attention score:
  `bias[h, i, j] = −m_h · |i − j|`, `m_h = 2^(−8h/H)`, h = 1..H
  - Bias tính trên toàn bộ L = 1 + n_win token, **kể cả CLS** (CLS được coi là vị trí 0)
  - Đối xứng (bidirectional), không causal
- `k3_layers` lớp encoder (mặc định **4**), `H = 4`, FFN = 256, dropout 0.1, GELU, pre-LN
- Lấy output CLS → `(B, D)`

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
  - 2 CLS token: trunc normal std = 0.02

---

## 8. Siêu tham số mặc định

| Tham số        | Giá trị | Tham số          | Giá trị |
|----------------|---------|------------------|---------|
| window W       | 64      | k2_layers        | 2       |
| stride St      | 32      | k2_heads / k2_ff | 4 / 256 |
| d_model D      | 64      | k3_layers        | 4       |
| feat (Conv out)| 64      | k3_heads / k3_ff | 4 / 256 |
| dropout        | 0.1     | activation       | GELU    |
| norm_first     | True    | sample_norm      | True    |
| k1_branch      | both    |                  |         |

**Training mặc định cho wiformer** (`main.py`): AdamW, lr = 1e-4, weight_decay = 0.01, batch = 32, warmup tuyến tính 5 epoch rồi CosineAnnealingLR (`T_max = epochs − warmup`).

---

## 9. Số tham số (UT_HAR, 7 lớp, cấu hình mặc định)

| Thành phần                          | Params      |
|-------------------------------------|-------------|
| Block 1 — Conv1d + BN               | 8,736       |
| Block 1 — proj_time                 | 4,160       |
| Block 1 — proj_freq                 | 2,112       |
| Block 1 — fuse                      | 8,256       |
| **Block 1 tổng**                    | **23,264**  |
| Block 2 — PE_sub (Embedding + Linear) | 3,072     |
| Block 2 — CLS                       | 64          |
| Block 2 — 2 × encoder layer (49,984/lớp) | 99,968 |
| **Block 2 tổng**                    | **103,104** |
| Block 3 — CLS                       | 64          |
| Block 3 — 4 × encoder layer         | 199,936     |
| **Block 3 tổng**                    | **200,000** |
| Classifier                          | 583         |
| **Toàn mô hình**                    | **326,951** |

Với sshar_esp (S = 56), Block 2 tăng lên 103,936 (Embedding lớn hơn) → tổng 327,783.

> Lưu ý: con số "~146K / ~161K" trong [CSI-WiFormer_Block1_FFT.md](CSI-WiFormer_Block1_FFT.md) không khớp với code vì code mặc định dùng k2_layers = 2, k3_layers = 4 và FFN = 256.
