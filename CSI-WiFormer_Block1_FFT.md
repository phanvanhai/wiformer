# Thay đổi Khối 1: Thêm nhánh FFT song song với Conv1d

## Lý do thiết kế

Tín hiệu CSI trong mỗi window W mẫu mang 2 loại thông tin trực giao:

- **Temporal pattern** (`v_time`): diễn biến hình dạng tín hiệu theo thời gian — Conv1d trích xuất hiệu quả nhưng phải suy diễn gián tiếp thông tin tần số.
- **Spectral pattern** (`v_freq`): cấu trúc tần số của tín hiệu — FFT cung cấp trực tiếp, xác định, không cần học.

Hai biểu diễn bổ sung cho nhau theo định lý Parseval: cùng lượng thông tin nhưng theo 2 góc nhìn trực giao, không cái nào bao hàm cái kia.

Hai vector được tích hợp qua `Linear(2D → D)` trước khi đưa vào Khối 2, đảm bảo mỗi token trong joint attention mang đầy đủ cả 2 loại đặc trưng.

---

## Thay đổi trong code

### File: `model/block1.py`

**Trước** — chỉ có nhánh Conv1d:

```python
class Block1(nn.Module):
    def __init__(self, D=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, D)

    def forward(self, x):
        # x: (B, N_rx, N_sub, T)
        B, N_rx, N_sub, T = x.shape

        x = x.unfold(-1, 64, 32)
        # x: (B, N_rx, N_sub, n_win=30, W=64)

        n_win = x.shape[3]
        x = x.reshape(B * N_rx * N_sub * n_win, 1, 64)
        x = self.encoder(x).squeeze(-1)   # (*, 64)
        x = self.proj(x)                   # (*, D)
        return x.reshape(B, N_rx, n_win, N_sub, D)
        # output: (B, N_rx, n_win=30, N_sub, D=64)
```

**Sau** — 2 nhánh song song trong cùng 1 Khối 1, fuse về D trước khi ra ngoài:

```python
class Block1(nn.Module):
    def __init__(self, D=64, W=64, S=32):
        super().__init__()
        self.W = W
        self.S = S

        # Nhánh 1: temporal — Conv1d stack (giữ nguyên)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj_time = nn.Linear(64, D)

        # Nhánh 2: spectral — FFT
        # rfft(W=64) → W//2+1 = 33 bin phức
        # Lấy biên độ, bỏ bin DC (bin 0) → 32 bin có nghĩa
        self.proj_freq = nn.Linear(W // 2, D)   # 32 → D

        # Fuse 2 nhánh: [v_time || v_freq] → D
        self.fuse = nn.Linear(2 * D, D)

    def forward(self, x):
        # x: (B, N_rx, N_sub, T)
        B, N_rx, N_sub, T = x.shape
        W, S = self.W, self.S

        # Unfold 1 lần — dùng chung cho cả 2 nhánh
        x_win = x.unfold(-1, W, S)
        # x_win: (B, N_rx, N_sub, n_win, W)
        n_win = x_win.shape[3]

        # Reshape để xử lý song song tất cả instance
        x_flat = x_win.reshape(B * N_rx * N_sub * n_win, W)
        # x_flat: (B·N_rx·N_sub·n_win, W)

        # --- Nhánh 1: temporal ---
        x_t = x_flat.unsqueeze(1)                # (*, 1, W)
        x_t = self.encoder(x_t).squeeze(-1)      # (*, 64)
        v_time = self.proj_time(x_t)              # (*, D)

        # --- Nhánh 2: spectral ---
        x_f = torch.fft.rfft(x_flat, dim=-1)     # (*, W//2+1) complex
        x_f = torch.log1p(x_f.abs())             # log-amplitude, (*, W//2+1)
        x_f = x_f[:, 1:]                          # bỏ bin DC → (*, W//2)
        v_freq = self.proj_freq(x_f)              # (*, D)

        # --- Fuse ---
        fused = torch.cat([v_time, v_freq], dim=-1)  # (*, 2D)
        out   = self.fuse(fused)                       # (*, D)

        return out.reshape(B, N_rx, n_win, N_sub, D)
        # output: (B, N_rx, n_win=30, N_sub, D=64) — giống hệt output cũ
```

---

## Những gì KHÔNG thay đổi

- Bước `unfold`: vẫn W=64, S=32, n_win=30 — **không đổi**
- Output shape Khối 1: vẫn `(B, N_rx, n_win=30, N_sub, D=64)` — **không đổi**
- `Block2` — joint (Rx × Sub) attention — **không đổi gì**
- `Block3` — Temporal Transformer — **không đổi gì**
- `Classifier` — **không đổi gì**
- Luồng tensor từ Khối 2 trở đi — **không đổi gì**

---

## Lưu ý kỹ thuật

**Tại sao dùng `log1p` thay vì raw amplitude**:

Phổ biên độ CSI thường có dynamic range lớn — bin tần số thấp (DC và gần DC) có biên độ rất lớn so với bin cao. `log1p(abs)` nén dynamic range, giúp gradient ổn định hơn khi huấn luyện:

```python
x_f = torch.log1p(x_f.abs())   # log(1 + |X(f)|)
# Thay vì: x_f = x_f.abs()
```

**Tại sao bỏ bin DC (bin 0)**:

Bin 0 là thành phần DC — giá trị trung bình của window, không mang thông tin về dao động. Sau z-score normalization, bin DC ≈ 0 với tất cả window → không có thông tin phân biệt. Bỏ đi để `proj_freq` tập trung vào 32 bin dao động có nghĩa.

**`W // 2` bin**:

`torch.fft.rfft(x, dim=-1)` với input độ dài W trả về `W//2 + 1` bin phức (0 đến W/2 Hz). Sau khi bỏ bin 0 (DC), còn lại `W//2` bin = 32 bin với W=64.

---

## Thống kê thay đổi

```
                        Trước           Sau             Chênh lệch
────────────────────────────────────────────────────────────────────
Params encoder          12,896          12,896          +0
Params proj             4,160           4,160  (proj_time)  +0
Params proj_freq        0               32×64+64=2,112  +2,112
Params fuse             0               2×64×64+64=8,256 +8,256
────────────────────────────────────────────────────────────────────
Tổng K1 params          12,896          27,424          +14,528

Unfold                  1 lần           1 lần           +0
FLOPs Conv1d            1.45G           1.45G           +0
FLOPs FFT               —               ~0.05G          +0.05G
────────────────────────────────────────────────────────────────────
Tổng K1 FLOPs           1.45G           1.50G           +3.4%

Khối 2 params           ~33K            ~33K            +0
Khối 3 params           ~100K           ~100K           +0
Seq Khối 3              31 token        31 token        +0
────────────────────────────────────────────────────────────────────
Tổng toàn mô hình       ~146K           ~161K           +14,528
```

---

## Import cần thêm

Không cần import thêm — `torch.fft` có sẵn trong PyTorch >= 1.7.

---

## Ablation study tự nhiên từ thiết kế này

Vì 2 nhánh tách biệt trong Khối 1, có thể ablate đơn giản bằng cách zero-out từng nhánh:

```python
# Variant A: chỉ temporal (baseline gốc)
v_freq = torch.zeros_like(v_freq)

# Variant B: chỉ spectral
v_time = torch.zeros_like(v_time)

# Variant C: cả 2 (đề xuất)
# giữ nguyên
```

Hoặc sạch hơn, truyền flag vào constructor:

```python
class Block1(nn.Module):
    def __init__(self, D=64, W=64, S=32,
                 use_time=True, use_freq=True):
        self.use_time = use_time
        self.use_freq = use_freq
        ...

    def forward(self, x):
        ...
        v_time = self.proj_time(x_t) if self.use_time \
                 else torch.zeros(batch, D, device=x.device)
        v_freq = self.proj_freq(x_f) if self.use_freq \
                 else torch.zeros(batch, D, device=x.device)
        ...
```
