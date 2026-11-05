"""
CSI-WiFormer (v4)

Input (from datasets.py)
------------------------
2d : (B, N_rx * N_ant * N_sub, T)     -- default loader output
3d : (B, N_rx * N_ant, N_sub, T)
4d : (B, N_rx, N_ant, N_sub, T)

Pipeline
--------
select antenna[0]  -> (B, N_rx, N_sub, T)
per-sample z-score -> (B, N_rx, N_sub, T)
Block 1 : unfold(W, S) + shared Conv1d stack -> (B, N_rx, N_sub, n_win, D)
Block 2 : joint (Rx x Sub) attention per window + CLS -> (B, n_win, D)
Block 3 : CLS + ALiBi temporal Transformer -> (B, D)
Classifier : LayerNorm -> Dropout -> Linear -> logits
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


# ============================================================
# Dataset configs: physical layout of the channel axis
# Channel order in datasets.py is (rx, antenna, sub) row-major.
# ============================================================
WIFORMER_DATASET_CFG = {
    # 3 RX x 1 antenna x 56 sub
    "sshar_esp": {"n_rx": 3, "n_ant": 1, "n_sub": 56},
    # 3 RX x 4 antenna x 56 sub -> keep antenna[0]
    "sshar_asus": {"n_rx": 3, "n_ant": 4, "n_sub": 56},
    # 3 RX x 3 antenna x 30 sub (270 channels) -> keep antenna[0]
    "xrf55": {"n_rx": 3, "n_ant": 3, "n_sub": 30},
    # 1 RX x 3 antenna x 30 sub: treat the 3 antennas as 3 views
    "ut_har": {"n_rx": 3, "n_ant": 1, "n_sub": 30},
}


# ============================================================
# Block 1 - Sub feature extractor
# ============================================================
class SubFeatureExtractor(nn.Module):
    """Shared Conv1d stack applied to every (rx, sub, window) of length W."""

    def __init__(self, d_model=64, feat=64, window=64, stride=32):
        super().__init__()
        self.window = window
        self.stride = stride
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=2, padding=3),     # W -> W/2
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, feat, kernel_size=4, stride=2, padding=1),  # W/2 -> W/4
            nn.BatchNorm1d(feat),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(feat, d_model)

    def forward(self, x):
        # x: (B, N_rx, N_sub, T)
        B, R, S, _ = x.shape
        x = x.unfold(dimension=-1, size=self.window, step=self.stride)
        # (B, R, S, n_win, W)
        n_win = x.shape[3]
        x = x.reshape(B * R * S * n_win, 1, self.window)
        x = self.encoder(x).squeeze(-1)          # (B*R*S*n_win, F)
        x = self.proj(x)                         # (B*R*S*n_win, D)
        return x.reshape(B, R, S, n_win, -1)     # (B, R, S, n_win, D)


# ============================================================
# Block 2 - Joint (Rx x Sub) attention per window
# ============================================================
class JointRxSubAttention(nn.Module):
    def __init__(
        self,
        n_rx,
        n_sub,
        d_model=64,
        n_heads=4,
        dim_ff=128,
        dropout=0.1,
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even (PE concat D/2 + D/2)"
        self.n_rx = n_rx
        self.n_sub = n_sub

        # Factorized PE: concat(PE_rx, PE_sub) -> Linear(D, D)
        self.embed_rx = nn.Embedding(n_rx, d_model // 2)
        self.embed_sub = nn.Embedding(n_sub, d_model // 2)
        self.pe_proj = nn.Linear(d_model, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.embed_rx.weight, std=0.02)
        nn.init.trunc_normal_(self.embed_sub.weight, std=0.02)

    def positional_encoding(self):
        device = self.embed_rx.weight.device
        pe_rx = self.embed_rx(torch.arange(self.n_rx, device=device))     # (R, D/2)
        pe_sub = self.embed_sub(torch.arange(self.n_sub, device=device))  # (S, D/2)
        pe = torch.cat(
            [
                pe_rx.unsqueeze(1).expand(-1, self.n_sub, -1),   # (R, S, D/2)
                pe_sub.unsqueeze(0).expand(self.n_rx, -1, -1),   # (R, S, D/2)
            ],
            dim=-1,
        ).reshape(self.n_rx * self.n_sub, -1)                    # (R*S, D)
        return self.pe_proj(pe)

    def forward(self, x):
        # x: (B, R, S, n_win, D)
        B, R, S, n_win, D = x.shape
        x = x.permute(0, 3, 1, 2, 4).reshape(B * n_win, R * S, D)
        x = x + self.positional_encoding().unsqueeze(0)

        cls = self.cls_token.expand(B * n_win, -1, -1)
        x = torch.cat([cls, x], dim=1)           # (B*n_win, 1+R*S, D)
        x = self.attn_layer(x)
        return x[:, 0].reshape(B, n_win, D)      # (B, n_win, D)


# ============================================================
# Block 3 - Temporal Transformer with ALiBi
# ============================================================
def get_alibi_slopes(n_heads):
    return 2 ** (-8 / n_heads * torch.arange(1, n_heads + 1, dtype=torch.float32))


class ALiBiEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with an additive attention bias."""

    def __init__(self, d_model=64, n_heads=4, dim_ff=256, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attn_bias):
        # x: (B, L, D) ; attn_bias: (1, H, L, L)
        B, L, D = x.shape
        h = self.norm1(x)
        q, k, v = (
            self.qkv(h)
            .reshape(B, L, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )                                        # each (B, H, L, hd)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias.to(q.dtype),
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                        # (B, H, L, hd)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        x = x + self.drop1(self.out_proj(attn))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList(
            [
                ALiBiEncoderLayer(d_model, n_heads, dim_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.register_buffer(
            "alibi_slopes", get_alibi_slopes(n_heads), persistent=False
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def build_alibi_bias(self, n_win, device):
        # bias(i, j) = -m_h * |i - j| between window tokens.
        # CLS (index 0) has no positional bias -> attends/is attended uniformly.
        pos = torch.arange(n_win, device=device)
        dist = (pos[None, :] - pos[:, None]).abs().float()        # (n_win, n_win)
        bias = torch.zeros(self.n_heads, n_win + 1, n_win + 1, device=device)
        bias[:, 1:, 1:] = -self.alibi_slopes.to(device)[:, None, None] * dist
        return bias.unsqueeze(0)                                  # (1, H, L, L)

    def forward(self, x):
        # x: (B, n_win, D)
        B, n_win, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)           # (B, 1+n_win, D)
        bias = self.build_alibi_bias(n_win, x.device)
        for layer in self.layers:
            x = layer(x, bias)
        return x[:, 0]                           # (B, D)


# ============================================================
# Full model
# ============================================================
class WiFormer(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        dataset=None,
        n_rx=None,
        n_ant=None,
        n_sub=None,
        window=64,
        stride=32,
        d_model=64,
        feat=64,
        k2_heads=4,
        k2_ff=128,
        k3_layers=2,
        k3_heads=4,
        k3_ff=256,
        dropout=0.1,
        sample_norm=True,
        **kwargs,
    ):
        super().__init__(input_shape=input_shape, num_classes=num_classes)

        cfg = dict(WIFORMER_DATASET_CFG.get((dataset or "").lower(), {}))
        n_rx = n_rx or cfg.get("n_rx")
        n_ant = n_ant or cfg.get("n_ant", 1)
        n_sub = n_sub or cfg.get("n_sub")
        if n_rx is None or n_sub is None:
            raise ValueError(
                f"WiFormer: unknown layout for dataset '{dataset}'. "
                "Pass n_rx / n_ant / n_sub explicitly."
            )
        self.n_rx, self.n_ant, self.n_sub = n_rx, n_ant, n_sub
        self.sample_norm = sample_norm

        # Validate against loader input shape
        T = input_shape[-1]
        n_channels = math.prod(input_shape[:-1])
        if n_channels != n_rx * n_ant * n_sub:
            raise ValueError(
                f"WiFormer: input_shape {tuple(input_shape)} has {n_channels} "
                f"channels, expected n_rx*n_ant*n_sub = {n_rx}*{n_ant}*{n_sub}"
            )
        if T < window:
            raise ValueError(f"WiFormer: T={T} < window={window}")
        self.n_win = (T - window) // stride + 1

        self.block1 = SubFeatureExtractor(d_model, feat, window, stride)
        self.block2 = JointRxSubAttention(
            n_rx, n_sub, d_model, k2_heads, k2_ff, dropout
        )
        self.block3 = TemporalTransformer(
            d_model, k3_heads, k3_layers, k3_ff, dropout
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def preprocess(self, x):
        # Any of (B, C, T) / (B, R*A, S, T) / (B, R, A, S, T) -> (B, R, S, T)
        B, T = x.shape[0], x.shape[-1]
        x = x.reshape(B, self.n_rx, self.n_ant, self.n_sub, T)
        x = x[:, :, 0]                           # antenna[0] of each RX
        if self.sample_norm:
            # z-score per (rx, sub) per sample, along time
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-6
            x = (x - mean) / std
        return x

    def forward(self, x):
        x = self.preprocess(x)                   # (B, R, S, T)
        x = self.block1(x)                       # (B, R, S, n_win, D)
        x = self.block2(x)                       # (B, n_win, D)
        x = self.block3(x)                       # (B, D)
        return self.classifier(x)                # (B, n_classes) logits
