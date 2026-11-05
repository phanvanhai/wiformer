# CSI-WiFormer: Kiến trúc nhận dạng hành động người với WiFi CSI
**Phiên bản**: v5 · N_rx: 3–6 · N_sub: 30 hoặc 56 · **Cập nhật**: tháng 11/2026

---

## 1. Tổng quan

CSI-WiFormer là kiến trúc học máy cho bài toán Human Activity Recognition (HAR) sử dụng tín hiệu WiFi CSI (Channel State Information). Kiến trúc được thiết kế dựa trên 3 insight vật lý:

1. **Subcarrier nhạy cảm thay đổi theo thời gian**: tại mỗi thời điểm, chỉ một số subcarrier "bắt được" chuyển động, tùy vào tần số, vị trí người, và hướng di chuyển. Mô hình cần học *sub nào đang nhạy* thay vì xử lý đồng đều.
2. **Hành động được tổ hợp từ micro-pattern**: hành động phức tạp là chuỗi có thứ tự của các primitive motion ngắn ~0.1–0.3s. Sliding window overlap đảm bảo mọi micro-pattern được biểu diễn nguyên vẹn trong ít nhất 1 window token, và tạo continuity giữa các token liền kề.
3. **Thông tin đa Rx bổ sung nhau**: sub_j của Rx_1 có thể cần thông tin từ sub_k của Rx_2 để phân biệt hành động — joint attention trên (Rx × Sub) đồng thời phản ánh đúng bản chất tín hiệu multipath đa Rx. Mô hình không dùng PE_rx (positional encoding theo index Rx) → mô hình học hoàn toàn từ nội dung tín hiệu, không từ identity của Rx.

---

## 2. Cấu hình hệ thống

| Thông số | Giá trị |
|---|---|
| Cấu hình anten | 1 Tx · N_rx Rx |
| Số anten sử dụng | **1 anten/Rx** — nếu Rx có nhiều anten, chỉ dùng anten[0] |
| N_rx | 3 đến 6 (linh hoạt theo dataset) |
| Sampling rate | 200 Hz |
| Độ dài mẫu | 5 giây → T = 1000 mẫu |
| N_sub | **30 hoặc 56** tùy dataset (xem mục 2.1) |
| Số lớp hành động | 7 |
| Các hành động | Đi, Chạy, Đứng lên, Ngồi xuống, Cúi nhặt, Ngã, Nằm im |

### 2.1 Lựa chọn anten và subcarrier theo dataset

```
Nếu Rx có nhiều anten:
  x = x[:, :, 0, :, :]   # chỉ lấy anten[0] của mỗi Rx
  → shape: (B, N_rx, T, N_sub)

N_sub theo chuẩn 802.11:
  802.11n 20 MHz → 56 subcarrier  (ví dụ: dataset tự thu, NTU-Fi HAR)
  802.11n 20 MHz → 30 subcarrier  (ví dụ: Widar3.0, UT-HAR)
```

Toàn bộ kiến trúc không phụ thuộc giá trị cụ thể của N_sub. Chỉ `Embedding(N_sub, D//2)` trong Khối 2 cần điều chỉnh.

### 2.2 Ảnh hưởng khi thay đổi N_rx và N_sub

| Khối | Thay đổi khi N_rx tăng | Thay đổi khi N_sub thay đổi |
|---|---|---|
| Khối 1 | FLOPs tăng tuyến tính · Params không đổi | FLOPs tăng tuyến tính · Params không đổi |
| Khối 2 | Seq tăng → FLOPs tăng O(N_rx²·N_sub²) · PE_sub không đổi | Seq thay đổi → FLOPs thay đổi · PE_sub thay đổi ~N_sub·32 params |
| Khối 3 | **Không đổi** | **Không đổi** |
| Classifier | **Không đổi** | **Không đổi** |

---

## 3. Siêu tham số

### 3.1 Siêu tham số cố định (kiến trúc)

| Ký hiệu | Giá trị | Mô tả |
|---|---|---|
| D | 64 | Chiều embedding toàn mô hình |
| F | 64 | Kênh đặc trưng output Conv1d stack (= D) |

**Khối 1 — Conv1d stack**:

| Ký hiệu | Giá trị | Mô tả |
|---|---|---|
| W | 64 mẫu = 0.32s | Window size — bắt được 1 chu kỳ chạy (0.25s), không loãng ngã (0.3–0.5s), lũy thừa 2 |
| S | 32 mẫu = 0.16s | Stride — overlap 50%, continuity giữa window liền kề |
| n_win | 30 | ⌊(T−W)/S⌋+1 = ⌊(1000−64)/32⌋+1 = 30 |
| k₁ | 8 | Kernel size lớp Conv1d thứ 1 |
| s₁ | 2 | Stride lớp Conv1d thứ 1 · W/2: 64→32 |
| C₁ | 32 | Số kênh output lớp Conv1d thứ 1 |
| k₂ | 4 | Kernel size lớp Conv1d thứ 2 |
| s₂ | 2 | Stride lớp Conv1d thứ 2 · 32→16 |
| C₂ | 64 | Số kênh output lớp Conv1d thứ 2 (= F) |
| RF | W=64 | Receptive field: lớp 1 nhìn 8 mẫu, lớp 2 nhìn 16 mẫu, AvgPool pool toàn 16 vị trí còn lại |

**Khối 2 — Joint (Rx × Sub) attention**:

| Ký hiệu | Giá trị | Mô tả |
|---|---|---|
| L₂ | 1 | Số TransformerEncoderLayer |
| h₂ | 4 | Số attention head |
| d_head₂ | 16 | Chiều per head (D/h₂ = 64/4) |
| FFN₂ | 256 | Chiều feedforward (4·D) |
| dropout₂ | 0.1 | Dropout trong attention và FFN |
| PE_rx | Không dùng | Không có positional encoding theo index Rx — mô hình học từ tín hiệu |
| PE_sub | Embedding(N_sub, D//2) | Encode thứ tự tần số của subcarrier |
| seq₂ | N_rx · N_sub + 1 | Sequence length (gồm CLS token) |

Ví dụ seq₂: N_rx=3, N_sub=56 → 169 token; N_rx=3, N_sub=30 → 91 token; N_rx=6, N_sub=56 → 337 token.

**Khối 3 — Temporal Transformer**:

| Ký hiệu | Giá trị | Mô tả |
|---|---|---|
| L₃ | 2 | Số TransformerEncoderLayer |
| h₃ | 4 | Số attention head |
| d_head₃ | 16 | Chiều per head (D/h₃ = 64/4) |
| FFN₃ | 256 | Chiều feedforward (4·D) |
| dropout₃ | 0.1 | Dropout trong attention và FFN |
| PE_win | ALiBi | Relative positional bias — không inject vị trí tuyệt đối |
| seq₃ | n_win + 1 = 31 | Sequence length (n_win=30 window token + 1 CLS token) |

ALiBi slope: `m_h = 2^(−8h/n_heads)` với h = 1..n_heads. Không có tham số học.

**Classifier**:

| Ký hiệu | Giá trị | Mô tả |
|---|---|---|
| dropout_clf | 0.1 | Dropout trước Linear |
| n_class | 7 | Số lớp hành động |

### 3.2 Siêu tham số huấn luyện (đề xuất)

| Ký hiệu | Giá trị đề xuất | Mô tả |
|---|---|---|
| Optimizer | AdamW | Weight decay = 0.01 |
| lr | 1e-4 | Learning rate khởi đầu |
| lr scheduler | CosineAnnealingLR | T_max = số epoch |
| batch size | 32 | Giảm xuống 16 nếu VRAM thiếu |
| epochs | 100 | Có thể dùng early stopping |
| warmup | 5 epoch | Linear warmup |
| weight init | Xavier uniform | Cho Linear; Conv dùng Kaiming |

---

## 4. Luồng tensor

```
Input raw      (B, N_rx, N_ant, T, N_sub)   nếu dataset có nhiều anten
  │
  ├─ Chọn anten  x = x[:, :, 0, :, :]
  │              (B, N_rx, T=1000, N_sub)   N_sub ∈ {30, 56}
  │
  ├─ Norm        z-score per sub per sample
  │              (B, N_rx, T, N_sub)
  │
  ├─ Khối 1     unfold(W=64, S=32) → Conv1d stack (shared weights)
  │              (B, N_rx, n_win=30, N_sub, D=64)
  │
  ├─ Khối 2     flatten (N_rx·N_sub token) + PE_sub + self-attn + CLS
  │              (B, n_win=30, D=64)   ← không phụ thuộc N_rx, N_sub
  │
  ├─ Khối 3     CLS + ALiBi + Transformer (31 token, L=2)
  │              (B, D=64)
  │
  └─ Classifier  LayerNorm → Dropout → Linear(D, n_class) → Softmax
                 (B, 7)
```

---

## 5. Kiến trúc chi tiết

### 5.0 Tiền xử lý

**Chọn anten** (nếu dataset có nhiều anten/Rx):

```python
# x: (B, N_rx, N_ant, T, N_sub)
x = x[:, :, 0, :, :]   # lấy anten[0] của mỗi Rx
# x: (B, N_rx, T, N_sub)
```

**Z-score normalization**:

```python
# x: (B, N_rx, T, N_sub)
mean = x.mean(dim=2, keepdim=True)   # mean trên trục T
std  = x.std(dim=2, keepdim=True) + 1e-6
x    = (x - mean) / std
```

---

### 5.1 Khối 1 — Sub feature extractor

**Params**: 12,896 (không đổi theo N_rx, N_sub)

```python
# Bước 1: Unfold
# x: (B, N_rx, N_sub, T)
x = x.unfold(dimension=-1, size=64, step=32)
# x: (B, N_rx, N_sub, n_win=30, W=64)

# Bước 2: Reshape để xử lý song song
x = x.permute(0, 1, 3, 2, 4)
x = x.reshape(B * N_rx * n_win * N_sub, 1, W)
# x: (B·N_rx·n_win·N_sub, 1, 64)

# Bước 3: Conv1d stack (shared weights)
# Conv1d(1→32, k=8, s=2) + BN + GELU   → (*, 32, 32)
# Conv1d(32→64, k=4, s=2) + BN + GELU  → (*, 64, 16)
# AdaptiveAvgPool1d(1)                   → (*, 64, 1)
# squeeze(-1)                            → (*, 64)
# Linear(64, D=64)                       → (*, D)

# Bước 4: Reshape về tensor gốc
x = x.reshape(B, N_rx, n_win, N_sub, D)
# x: (B, N_rx, n_win=30, N_sub, D=64)
```

---

### 5.2 Khối 2 — Joint (Rx × Sub) attention per window

**Params**: D²/h₂ · 4 · h₂ + 2·FFN₂·D + N_sub·D//2 + D² + D ≈ 33K + N_sub·32

Không dùng PE_rx — mô hình không biết token đến từ Rx nào, chỉ biết token đến từ subcarrier nào (qua PE_sub). Attention phân biệt các Rx hoàn toàn từ nội dung tín hiệu.

```python
# x: (B, N_rx, n_win, N_sub, D)

# Bước 1: Tách per window, flatten Rx và Sub
x = x.permute(0, 2, 1, 3, 4)
x = x.reshape(B * n_win, N_rx * N_sub, D)
# x: (B·n_win, N_rx·N_sub, D)

# Bước 2: Positional encoding — chỉ PE_sub, không PE_rx
sub_idx = torch.arange(N_sub, device=x.device)
pe_sub  = self.embed_sub(sub_idx)                    # (N_sub, D//2)
# Project D//2 → D, broadcast cho N_rx
pe = self.pe_proj(
    pe_sub.unsqueeze(0).expand(N_rx, -1, -1)         # (N_rx, N_sub, D//2)
         .reshape(N_rx * N_sub, D // 2)
)                                                      # (N_rx·N_sub, D)
x = x + pe.unsqueeze(0)                              # broadcast B·n_win

# Bước 3: Prepend CLS token
cls = self.cls_token.expand(B * n_win, 1, D)
x   = torch.cat([cls, x], dim=1)
# x: (B·n_win, 1 + N_rx·N_sub, D)

# Bước 4: Self-attention (1 layer)
x = self.attn_layer(x)   # TransformerEncoderLayer(D, h₂=4, FFN₂=256)

# Bước 5: Lấy CLS output
out = x[:, 0, :].reshape(B, n_win, D)
# out: (B, n_win=30, D=64)
```

---

### 5.3 Khối 3 — Temporal Transformer

**Params**: L₃ · (4·D² + 2·FFN₃·D + 4·D) ≈ 2 · (16K + 32K + 256) ≈ 100K

```python
# x: (B, n_win=30, D)

# Bước 1: ALiBi bias matrix (không có tham số học)
slopes = 2 ** (-8 / h₃ * torch.arange(1, h₃+1))   # (h₃,)
dist   = torch.abs(
    torch.arange(seq₃).unsqueeze(0) -
    torch.arange(seq₃).unsqueeze(1)
)                                                     # (seq₃, seq₃)
alibi  = -slopes.view(-1,1,1) * dist.unsqueeze(0)   # (h₃, seq₃, seq₃)

# Bước 2: Prepend CLS token
cls = self.cls_token.expand(B, 1, D)
x   = torch.cat([cls, x], dim=1)
# x: (B, 31, D)

# Bước 3: Transformer Encoder (L₃=2 layers)
x = self.transformer_encoder(x, attn_bias=alibi)
# x: (B, 31, D)

# Bước 4: Lấy CLS output
out = x[:, 0, :]
# out: (B, D=64)
```

---

### 5.4 Classifier

**Params**: D·n_class + n_class + 2·D = 64·7 + 7 + 128 = 583

```python
# x: (B, D=64)
x = nn.LayerNorm(D)(x)
x = nn.Dropout(0.1)(x)
x = nn.Linear(D, n_class)(x)   # (B, 7)
# → Softmax → P(action)
```

---

## 6. Thống kê tham số và FLOPs

### 6.1 Số tham số chi tiết

```
─────────────────────────────────────────────────────────────
Khối 0  Norm                             0
─────────────────────────────────────────────────────────────
Khối 1  Conv1d(1→32, k=8)               1×32×8 + 32   =    288
        BN(32)                           32×2          =     64
        Conv1d(32→64, k=4)              32×64×4 + 64  =  8,256
        BN(64)                           64×2          =    128
        Linear(64, D=64)                64×64 + 64    =  4,160
        ─────────────────────────────────────────────── 12,896
─────────────────────────────────────────────────────────────
Khối 2  PE_sub  Embedding(N_sub, D//2)
                N_sub=56: 56×32         =  1,792
                N_sub=30: 30×32         =    960
        pe_proj Linear(D//2, D)         32×64+64      =  2,112
        CLS token                       1×D           =     64
        MHA     W_Q,K,V,O (4×D²/h₂×h₂) 4×D²          = 16,384
        FFN     W₁,b₁,W₂,b₂            2×D×FFN₂      = 32,768
        LayerNorm ×2                    4×D            =    256
        ─────────────────────────────────────────────────────
        Tổng K2 (N_sub=56)                            ~53,376
        Tổng K2 (N_sub=30)                            ~52,544
─────────────────────────────────────────────────────────────
Khối 3  CLS token                       1×D           =     64
        MHA ×2 layers                   2×4×D²        = 32,768
        FFN ×2 layers                   2×2×D×FFN₃    = 65,536
        LayerNorm ×4                    4×2×D          =    512
        ─────────────────────────────────────────────── 98,880
─────────────────────────────────────────────────────────────
Classifier
        LayerNorm                       2×D            =    128
        Linear(D, 7)                    D×7+7          =    455
        ─────────────────────────────────────────────────  583
─────────────────────────────────────────────────────────────
TỔNG (N_sub=56, N_rx=3)                              ~165,735
TỔNG (N_sub=30, N_rx=3)                              ~164,903
─────────────────────────────────────────────────────────────
```

*Lưu ý: N_rx không ảnh hưởng đến số tham số vì không có PE_rx.*

### 6.2 FLOPs ước tính per sample

```
─────────────────────────────────────────────────────────────────────
Khối 1  instances = N_rx × N_sub × n_win
        N_rx=3, N_sub=56: 3×56×30 = 5,040 instances
        FLOPs/instance:
          Conv1d lớp 1:  2×8×1×32×32    =   16,384
          Conv1d lớp 2:  2×4×32×64×16   =  262,144
          AvgPool:        64×16          =    1,024
          Linear:         2×64×64        =    8,192
        FLOPs/instance ≈ 287,744 ≈ 288K
        Tổng K1 (N_rx=3, N_sub=56): 5,040×288K ≈ 1.45G
        Tổng K1 (N_rx=3, N_sub=30): 2,700×288K ≈ 0.78G

Khối 2  seq = N_rx·N_sub + 1
        FLOPs attention ≈ 4·seq²·D + 2·seq·D·FFN₂
        N_rx=3, N_sub=56: seq=169
          attn: 4×169²×64 ≈ 7.3M · FFN: 2×169×64×256 ≈ 5.6M
          per win ≈ 12.9M · ×30 win ≈ 387M per sample
        N_rx=3, N_sub=30: seq=91
          per win ≈ 3.9M  · ×30 win ≈ 117M per sample

Khối 3  seq=31, L=2
        attn: 2×4×31²×64 ≈ 0.49M
        FFN:  2×2×31×64×256 ≈ 2.05M
        Tổng K3 ≈ 2.5M per sample

─────────────────────────────────────────────────────────────────────
TỔNG FLOPs/sample (N_rx=3, N_sub=56) ≈ 1.45G + 0.39G + 0.003G ≈ 1.84G
TỔNG FLOPs/sample (N_rx=3, N_sub=30) ≈ 0.78G + 0.12G + 0.003G ≈ 0.90G
TỔNG FLOPs/sample (N_rx=6, N_sub=56) ≈ 2.90G + 1.46G + 0.003G ≈ 4.36G
─────────────────────────────────────────────────────────────────────
```

### 6.3 So sánh tổng quan

```
M� hình              Params      FLOPs/sample (ước tính)
──────────────────────────────────────────────────────────
THAT (AAAI'21)       ~2M         ~1.8G
WiFlexFormer (2024)  ~100K       ~0.3G
AMCS-TCN             ~2M+        ~2G+
UniFi (IMWUT'24)     ~11M+       ~5G+
CSI-WiFormer N_sub=56 ~166K      ~1.84G
CSI-WiFormer N_sub=30 ~165K      ~0.90G
```

---

## 7. So sánh với các phương pháp liên quan

### 7.1 Transformer-based (baseline chính)

| Tiêu chí | THAT (AAAI'21) | WiFlexFormer (2024) | AMCS-TCN | **CSI-WiFormer** |
|---|---|---|---|---|
| Input | Raw CSI | Raw / DFS | Raw CSI | **Raw CSI** |
| Tokenization | 2 stream song song | Stem + 1D seq | Multi-scale conv | **Sliding win overlap 50%** |
| Sub selection | Gaussian PE | Không | Scale selection | **Joint per window** |
| Cross Rx × Sub | Không | Không | Không (fuse sớm) | **Trực tiếp** |
| Sliding overlap | Không | Không | Không | **Có** |
| Multi-Rx | Không | Không | Fuse global | **Joint, N_rx linh hoạt** |
| Positional encoding | Gaussian range | Standard APE | — | **ALiBi (relative)** |
| Params | ~2M | ~100K | ~2M+ | **~166K** |
| Code công khai | Không | Có | Không | — |

### 7.2 Multi-view attention (tham khảo)

| Tiêu chí | UniFi (IMWUT'24) | **CSI-WiFormer** |
|---|---|---|
| Task | Nhận dạng cử chỉ | HAR thô |
| Input | Spectrogram (ảnh RGB) | Raw CSI |
| Cross Rx × Sub | Per timestep t độc lập | **Joint đồng thời** |
| Cần vị trí Rx | Có (Fresnel zone mask) | **Không** |
| Mục tiêu | Cross-domain generalization | Classification |

---

## 8. Điểm thiết kế nổi bật

**Joint (Rx × Sub) attention** (Khối 2): [Rx_i, sub_j] attend trực tiếp đến [Rx_k, sub_l] trong 1 operation — không factorized như THAT hay UniFi.

**Không dùng PE_rx**: mô hình không biết token đến từ Rx nào — hoàn toàn học từ nội dung tín hiệu. Phân biệt Rx hoàn toàn qua attention score. Đúng về mặt vật lý khi người đứng ở vị trí bất kỳ trong phòng.

**Chỉ dùng PE_sub**: subcarrier có cấu trúc tần số — sub liền kề tương quan cao hơn sub xa. PE_sub encode thứ tự tần số này mà không tạo shortcut theo identity Rx.

**Sliding window overlap 50%** (Khối 1): đảm bảo micro-pattern không bị cắt đứt giữa 2 window; tạo continuity trong chuỗi token.

**ALiBi positional bias** (Khối 3): encode khoảng cách tương đối, không vị trí tuyệt đối → không bị bias khi hành động xuất hiện ở đầu/giữa/cuối mẫu 5s.

**Scale tự nhiên theo N_rx và N_sub**: thay đổi N_rx không ảnh hưởng params (không có PE_rx); thay đổi N_sub chỉ ảnh hưởng PE_sub (~32 params/sub); Khối 3 và Classifier hoàn toàn không thay đổi.

**Shared weights** (Khối 1 và Khối 2): ~166K params tổng — nhỏ hơn 10–70× so với baseline.

---

*Tài liệu mô tả kiến trúc tại thời điểm thiết kế. Hyperparameter cần xác nhận qua thực nghiệm.*
