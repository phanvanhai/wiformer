# CSI-WiFormer: Kiến trúc nhận dạng hành động người với WiFi CSI
**Phiên bản**: v4 · N_rx: 3–6 · N_sub: 30 hoặc 56 · **Cập nhật**: tháng 11/2026

---

## 1. Tổng quan

CSI-WiFormer là kiến trúc học máy cho bài toán Human Activity Recognition (HAR) sử dụng tín hiệu WiFi CSI (Channel State Information). Kiến trúc được thiết kế dựa trên 3 insight vật lý:

1. **Subcarrier nhạy cảm thay đổi theo thời gian**: tại mỗi thời điểm, chỉ một số subcarrier "bắt được" chuyển động, tùy vào tần số, vị trí người, và hướng di chuyển. Mô hình cần học *sub nào đang nhạy* thay vì xử lý đồng đều.
2. **Hành động được tổ hợp từ micro-pattern**: hành động phức tạp là chuỗi có thứ tự của các primitive motion ngắn ~0.1–0.3s. Sliding window overlap đảm bảo mọi micro-pattern được biểu diễn nguyên vẹn trong ít nhất 1 window token, và tạo continuity giữa các token liền kề.
3. **Thông tin đa Rx bổ sung nhau**: sub_j của Rx_1 có thể cần thông tin từ sub_k của Rx_2 để phân biệt hành động — joint attention trên (Rx × Sub) đồng thời phản ánh đúng bản chất tín hiệu multipath đa Rx. Mô hình không dùng prior vị trí vật lý của Rx → permutation-invariant → đúng khi người đứng ở các vị trí khác nhau.

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
| Số lớp | 7 hành động |
| Các hành động | Đi, Chạy, Đứng lên, Ngồi xuống, Cúi nhặt, Ngã, Nằm im |

### 2.1 Lựa chọn anten và subcarrier theo dataset

Nhiều dataset WiFi CSI thu thập tín hiệu với nhiều anten mỗi Rx và số lượng subcarrier khác nhau tùy chuẩn 802.11:

```
Nếu Rx có nhiều anten:
  x = x[:, :, 0, :, :]   # chỉ lấy anten[0] của mỗi Rx
  → shape: (B, N_rx, T, N_sub)

N_sub theo dataset:
  802.11n  20 MHz → 56 subcarrier (ví dụ: dataset tự thu, NTU-Fi)
  802.11n  20 MHz → 30 subcarrier (ví dụ: Widar3.0, UT-HAR)
  → N_sub là hyperparameter, không ảnh hưởng kiến trúc
```

Toàn bộ kiến trúc (Khối 1, 2, 3) **không phụ thuộc giá trị cụ thể của N_sub** — chỉ PE_sub trong Khối 2 cần điều chỉnh `Embedding(N_sub, D//2)`.

### 2.2 Ảnh hưởng khi thay đổi N_rx (3→6)

| Khối | Thay đổi |
|---|---|
| Khối 1 | FLOPs tăng tuyến tính 2× · Params không đổi (shared weights) |
| Khối 2 | Seq tăng: N_rx·N_sub token · FLOPs tăng O(N_rx²) · Params tăng 32/Rx (PE_rx) |
| Khối 3 | **Không đổi** — Khối 2 đã pool về 1 vector/window bất kể N_rx |
| Classifier | **Không đổi** |

Ví dụ N_sub=56: N_rx=3 → 168 token; N_rx=6 → 336 token. Với Flash Attention (PyTorch ≥ 2.0), 336 token chạy được với VRAM 8GB+.

---

## 3. Hyperparameter

| Ký hiệu | Giá trị | Lý do |
|---|---|---|
| W | 64 mẫu = 0.32s | Bắt được 1 chu kỳ chạy (0.25s), không loãng ngã (0.3–0.5s), lũy thừa 2 |
| S | 32 mẫu = 0.16s | Overlap 50% — continuity giữa window liền kề |
| n_win | 30 | ⌊(1000−64)/32⌋+1 = 30 |
| F | 64 | Kênh đặc trưng Conv1d stack |
| D | 64 | Chiều embedding toàn mô hình |
| K2 n_heads | 4 | Multi-head attention Khối 2 |
| K3 L | 2 | Số Transformer layer |
| K3 n_heads | 4 | Số attention head |
| K3 FFN | 256 | Chiều feedforward |
| dropout | 0.1 | Áp dụng Transformer và Classifier |

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
  ├─ Khối 1     unfold(W=64, S=32) → Conv1d stack (shared)
  │              (B, N_rx, n_win=30, N_sub, D=64)
  │
  ├─ Khối 2     flatten (N_rx×N_sub token) + PE + self-attn + CLS
  │              (B, n_win=30, D=64)   ← không phụ thuộc N_rx, N_sub
  │
  ├─ Khối 3     CLS + ALiBi + Transformer (31 token, L=2)
  │              (B, D=64)
  │
  └─ Classifier  LayerNorm → Dropout → Linear(D, n_class) → Softmax
                 (B, n_class)
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
mean = x.mean(dim=2, keepdim=True)
std  = x.std(dim=2, keepdim=True) + 1e-6
x    = (x - mean) / std
```

M��c đích: loại bỏ bias biên độ tuyệt đối phụ thuộc môi trường và thiết bị. Mô hình chỉ học pattern tương đối của chuyển động.

---

### 5.1 Khối 1 — Sub feature extractor

M��c đích: trích xuất primitive motion pattern trong từng window 0.32s của từng subcarrier. Shared weights giữa tất cả subcarrier, Rx, và window.

**Bước 1 — Unfold (sliding window)**:

```python
# x: (B, N_rx, N_sub, T)
x = x.unfold(dimension=-1, size=64, step=32)
# x: (B, N_rx, N_sub, n_win=30, W=64)
```

Overlap 50%: hai window liền kề chia sẻ 32 mẫu → continuity trong chuỗi token đưa vào Khối 3. Mọi micro-pattern xuất hiện nguyên vẹn trong ít nhất 1 window.

**Bước 2 — Conv1d stack (shared weights)**:

```python
# Reshape: (B·N_rx·N_sub·n_win, 1, W=64)
encoder = nn.Sequential(
    nn.Conv1d(1,  32, kernel_size=8, stride=2, padding=3),  # → (*, 32, 32)
    nn.BatchNorm1d(32),
    nn.GELU(),
    nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),  # → (*, 64, 16)
    nn.BatchNorm1d(64),
    nn.GELU(),
    nn.AdaptiveAvgPool1d(1),                                  # → (*, 64, 1)
)
# → squeeze → Linear(64, D=64)
# → reshape → (B, N_rx, n_win=30, N_sub, D=64)
```

Receptive field: lớp 1 nhìn 8 mẫu, lớp 2 nhìn 16 mẫu, AvgPool pool toàn bộ W=64 → phủ đủ window.

**Output**: `(B, N_rx, n_win=30, N_sub, D=64)` · **Params**: 12,896 (không đổi theo N_rx, N_sub)

---

### 5.2 Khối 2 — Joint (Rx × Sub) attention per window

M��c đích: học đồng thời "subcarrier nào và Rx nào đang nhạy cảm trong window này". [Rx_i, sub_j] attend trực tiếp đến [Rx_k, sub_l] trong 1 bước — không factorized.

```python
# x: (B, N_rx, n_win, N_sub, D)
# Xử lý độc lập cho mỗi window
x = x.permute(0, 2, 1, 3, 4)
x = x.reshape(B * n_win, N_rx * N_sub, D)   # (B·n_win, N_rx·N_sub, D)

# Factorized positional encoding: concat thay vì cộng
# → phân biệt (Rx2,sub3) với (Rx3,sub2)
pe_rx  = self.embed_rx(rx_idx)    # Embedding(N_rx,  D//2)
pe_sub = self.embed_sub(sub_idx)  # Embedding(N_sub, D//2)
pe = torch.cat([
    pe_rx.unsqueeze(1).expand(-1, N_sub, -1),   # (N_rx, N_sub, D//2)
    pe_sub.unsqueeze(0).expand(N_rx, -1, -1),   # (N_rx, N_sub, D//2)
], dim=-1).reshape(N_rx * N_sub, D)              # (N_rx·N_sub, D)
pe = self.pe_proj(pe)                            # Linear(D, D)
x  = x + pe.unsqueeze(0)                        # broadcast over B·n_win

# Prepend CLS token + self-attention
cls = self.cls_token.expand(B * n_win, 1, D)
x   = torch.cat([cls, x], dim=1)                # (B·n_win, 1+N_rx·N_sub, D)
x   = self.attn_layer(x)                        # TransformerEncoderLayer, n_heads=4

# Lấy CLS output
out = x[:, 0, :].reshape(B, n_win, D)           # (B, n_win=30, D)
```

Lý do không dùng PE_rx theo vị trí vật lý: mô hình học từ tín hiệu thuần túy → permutation-invariant trên trục Rx → đúng khi người đứng gần các Rx khác nhau trong phòng.

**Output**: `(B, n_win=30, D=64)` · **Params**: ~33K + 32·N_rx (PE_rx)

---

### 5.3 Khối 3 — Temporal Transformer

M��c đích: học temporal composition — cách các primitive pattern (window token) kết hợp thành sub-action và action. Window liền kề chia sẻ 50% tín hiệu → Transformer học transition giữa các pattern.

**ALiBi positional bias**:

```python
# Không inject vị trí tuyệt đối → không bias khi hành động
# xuất hiện ở đầu, giữa, hay cuối mẫu 5s
# score(i,j) = Q_i · K_j / √D − m · |i−j|
# m: slope cố định per head, không học

def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    return 2 ** (-8 / n_heads * torch.arange(1, n_heads + 1, dtype=torch.float32))
```

**Pipeline**:

```python
# x: (B, n_win=30, D)
cls = self.cls_token.expand(B, 1, D)
x   = torch.cat([cls, x], dim=1)              # (B, 31, D)
x   = self.transformer_encoder(x,
          attn_bias=alibi_bias)                # L=2, n_heads=4, FFN=256, dropout=0.1
out = x[:, 0, :]                               # CLS → (B, D=64)
```

**Output**: `(B, D=64)` · **Params**: ~100K · Seq=31 token · Không đổi theo N_rx, N_sub

---

### 5.4 Classifier

```python
classifier = nn.Sequential(
    nn.LayerNorm(D),
    nn.Dropout(0.1),
    nn.Linear(D, n_class),
)
# → Softmax → P(action)
```

**Params**: D·n_class + n_class + 2D (ví dụ: 7 lớp → 583 params)

---

## 6. Thống kê mô hình

```
Cấu hình: N_rx=3, N_sub=56, D=64, n_class=7

Khối           Params          FLOPs/sample
──────────────────────────────────────────────
Khối 1         12,896          ~1.45G
Khối 2         ~33,376         ~2.47G
Khối 3         ~99,992         ~4.1M
Classifier     583             <0.1M
──────────────────────────────────────────────
Tổng           ~146,847        ~3.93G

Với N_sub=30 (thay vì 56), Khối 2:
  Seq: 3×30=90 token (thay vì 168)
  FLOPs Khối 2 giảm ~3.5×
  Params Khối 2 giảm ~1K (PE_sub nhỏ hơn)

Với N_rx=6 (thay vì 3):
  FLOPs Khối 1 tăng 2×
  FLOPs Khối 2 tăng ~4× (do seq² tăng 4×)
  Params tăng thêm 3×32=96
  Khối 3 không thay đổi
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
| Params | ~2M | ~100K | ~2M+ | **~147K** |
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

**Joint (Rx × Sub) attention** (Khối 2): [Rx_i, sub_j] attend trực tiếp đến [Rx_k, sub_l] trong 1 operation — không factorized như THAT hay UniFi. Phản ánh bản chất vật lý: sub_j của Rx_1 cần thông tin từ sub_k của Rx_2.

**Sliding window overlap 50%** (Khối 1): đảm bảo micro-pattern không bị cắt đứt giữa 2 window; tạo continuity trong chuỗi token (window liền kề chia sẻ 50% tín hiệu).

**Permutation-invariant trên Rx**: không dùng PE_rx theo vị trí vật lý → mô hình học từ tín hiệu thuần túy → đúng khi người đứng ở vị trí bất kỳ.

**ALiBi positional bias** (Khối 3): encode khoảng cách tương đối, không vị trí tuyệt đối → không bị bias khi hành động xuất hiện ở đầu/giữa/cuối mẫu 5s.

**Scale tự nhiên theo N_rx và N_sub**: thêm Rx chỉ tăng thêm 32 params/Rx (PE_rx); thay đổi N_sub chỉ ảnh hưởng PE_sub; Khối 3 và Classifier hoàn toàn không thay đổi.

**Shared weights** (Khối 1 và Khối 2): ~147K params — nhỏ hơn 10–70× so với các baseline.

---

*Tài liệu mô tả kiến trúc tại thời điểm thiết kế. Hyperparameter cần xác nhận qua thực nghiệm.*
