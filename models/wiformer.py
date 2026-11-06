"""
CSI-WiFormer (v5) - see WiFormer_Architecture.md

Input (from datasets.py)
------------------------
2d : (B, N_rx * N_ant * N_sub, T)     -- default loader output
3d : (B, N_rx * N_ant, N_sub, T)
4d : (B, N_rx, N_ant, N_sub, T)

Pipeline
--------
select antenna[0]  -> (B, N_rx, N_sub, T)
per-sample z-score -> (B, N_rx, N_sub, T)
Block 1 : unfold(W, S) + [Conv1d branch || FFT branch] -> fuse -> (B, N_rx, n_win, N_sub, D)
Block 2 : joint (Rx x Sub) attention per window, PE_sub only, CLS -> (B, n_win, D)
Block 3 : CLS + ALiBi temporal Transformer -> (B, D)
Classifier : LayerNorm -> Dropout -> Linear -> logits
             (Softmax is applied inside CrossEntropyLoss / at inference)
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


class SafeTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    nn.TransformerEncoderLayer that never takes PyTorch's eval-mode "fast path".

    With a per-head float mask (ALiBi, shape (B*H, L, L)) the fused fast path
    (torch._transformer_encoder_layer_fwd / _native_multi_head_attention)
    returns the same output for every sample in eval mode -> constant
    predictions at test time (acc = 1/num_classes). Same parameters /
    state_dict keys as the parent class.
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._safe_sa(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._safe_sa(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _safe_sa(self, x, attn_mask, key_padding_mask):
        # Call the functional MHA directly so MultiheadAttention's own
        # fast path is bypassed as well.
        attn = self.self_attn
        q = x.transpose(0, 1)                              # (L, B, D)
        out, _ = F.multi_head_attention_forward(
            q, q, q,
            attn.embed_dim, attn.num_heads,
            attn.in_proj_weight, attn.in_proj_bias,
            attn.bias_k, attn.bias_v, attn.add_zero_attn,
            attn.dropout, attn.out_proj.weight, attn.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        return self.dropout1(out.transpose(0, 1))


def make_encoder_layer(d_model, n_heads, dim_ff, dropout, activation, norm_first):
    return SafeTransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=dim_ff,
        dropout=dropout,
        activation=activation,
        batch_first=True,
        norm_first=norm_first,
    )


# ============================================================
# Block 1 - Sub feature extractor
# ============================================================
class SubFeatureExtractor(nn.Module):
    """
    Two parallel branches on every (rx, window, sub) segment of length W,
    fused back to D (see CSI-WiFormer_Block1_FFT.md):

    - temporal : shared Conv1d stack -> Linear(feat, D)          -> v_time
    - spectral : rfft -> log1p|X| -> drop DC -> Linear(W//2, D)   -> v_freq
    - fuse     : Linear(2D, D) on [v_time || v_freq]

    use_time / use_freq allow ablation (a disabled branch contributes zeros).
    """

    def __init__(self, d_model=64, feat=64, window=64, stride=32,
                 use_time=True, use_freq=True):
        super().__init__()
        if not (use_time or use_freq):
            raise ValueError("SubFeatureExtractor: enable at least one branch")
        self.window = window
        self.stride = stride
        self.d_model = d_model
        self.use_time = use_time
        self.use_freq = use_freq

        # Branch 1: temporal (Conv1d stack)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=2, padding=3),     # 64 -> 32
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, feat, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm1d(feat),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),                                  # 16 -> 1
        )
        self.proj_time = nn.Linear(feat, d_model)

        # Branch 2: spectral (FFT). rfft(W) -> W//2+1 bins, drop DC -> W//2
        self.proj_freq = nn.Linear(window // 2, d_model)

        # Fuse [v_time || v_freq] -> D
        self.fuse = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        # x: (B, N_rx, N_sub, T)
        B, R, S, _ = x.shape
        x = x.unfold(dimension=-1, size=self.window, step=self.stride)
        # (B, R, S, n_win, W)
        n_win = x.shape[3]
        x = x.permute(0, 1, 3, 2, 4)                       # (B, R, n_win, S, W)
        x = x.reshape(B * R * n_win * S, self.window)      # (N, W), shared by both branches
        N = x.shape[0]

        # --- temporal ---
        if self.use_time:
            v_time = self.encoder(x.unsqueeze(1)).squeeze(-1)  # (N, F)
            v_time = self.proj_time(v_time)                    # (N, D)
        else:
            v_time = x.new_zeros(N, self.d_model)

        # --- spectral ---
        if self.use_freq:
            # FFT in float32 (rfft does not support half on all backends)
            x_f = torch.fft.rfft(x.float(), dim=-1)            # (N, W//2+1) complex
            x_f = torch.log1p(x_f.abs())[:, 1:].to(x.dtype)    # drop DC -> (N, W//2)
            v_freq = self.proj_freq(x_f)                       # (N, D)
        else:
            v_freq = x.new_zeros(N, self.d_model)

        # --- fuse ---
        out = self.fuse(torch.cat([v_time, v_freq], dim=-1))  # (N, D)
        return out.reshape(B, R, n_win, S, -1)               # (B, R, n_win, S, D)


# ============================================================
# Block 2 - Joint (Rx x Sub) attention per window (PE_sub only)
# ============================================================
class JointRxSubAttention(nn.Module):
    def __init__(
        self,
        n_sub,
        d_model=64,
        n_heads=4,
        n_layers=1,
        dim_ff=256,
        dropout=0.1,
        activation="gelu",
        norm_first=True,
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.n_sub = n_sub

        # PE_sub: Embedding(N_sub, D/2) -> Linear(D/2, D). No PE_rx.
        self.embed_sub = nn.Embedding(n_sub, d_model // 2)
        self.pe_proj = nn.Linear(d_model // 2, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn_layer = nn.TransformerEncoder(
            make_encoder_layer(
                d_model, n_heads, dim_ff, dropout, activation, norm_first
            ),
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x):
        # x: (B, R, n_win, S, D)
        B, R, n_win, S, D = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * n_win, R * S, D)

        sub_idx = torch.arange(S, device=x.device)
        pe = self.pe_proj(self.embed_sub(sub_idx))          # (S, D)
        pe = pe.unsqueeze(0).expand(R, -1, -1).reshape(R * S, D)
        x = x + pe.unsqueeze(0)                             # broadcast B*n_win

        cls = self.cls_token.expand(B * n_win, -1, -1)
        x = torch.cat([cls, x], dim=1)                      # (B*n_win, 1+R*S, D)
        x = self.attn_layer(x)
        return x[:, 0].reshape(B, n_win, D)                 # (B, n_win, D)


# ============================================================
# Block 3 - Temporal Transformer with ALiBi
# ============================================================
def get_alibi_slopes(n_heads):
    return 2 ** (-8 / n_heads * torch.arange(1, n_heads + 1, dtype=torch.float32))


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_ff=256,
        dropout=0.1,
        activation="gelu",
        norm_first=True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.encoder = nn.TransformerEncoder(
            make_encoder_layer(
                d_model, n_heads, dim_ff, dropout, activation, norm_first
            ),
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.register_buffer(
            "alibi_slopes", get_alibi_slopes(n_heads), persistent=False
        )

    def build_alibi_bias(self, seq_len, device, dtype):
        # alibi[h, i, j] = -m_h * |i - j| over all seq_len = n_win + 1 tokens
        pos = torch.arange(seq_len, device=device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().to(dtype)
        return -self.alibi_slopes.to(device, dtype).view(-1, 1, 1) * dist.unsqueeze(0)

    def forward(self, x):
        # x: (B, n_win, D)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                      # (B, 1+n_win, D)
        L = x.shape[1]

        alibi = self.build_alibi_bias(L, x.device, x.dtype)  # (H, L, L)
        # nn.MultiheadAttention float mask: (B*H, L, L), batch-major
        mask = alibi.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.n_heads, L, L)

        x = self.encoder(x, mask=mask)
        return x[:, 0]                                      # (B, D)


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
        k2_layers=2,
        k2_heads=4,
        k2_ff=256,
        k3_layers=4,
        k3_heads=4,
        k3_ff=256,
        dropout=0.1,
        activation="gelu",
        norm_first=True,
        sample_norm=True,
        k1_branch="both",
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

        if k1_branch not in ("both", "time", "freq"):
            raise ValueError(f"WiFormer: k1_branch must be both/time/freq, got '{k1_branch}'")
        self.block1 = SubFeatureExtractor(
            d_model, feat, window, stride,
            use_time=k1_branch in ("both", "time"),
            use_freq=k1_branch in ("both", "freq"),
        )
        self.block2 = JointRxSubAttention(
            n_sub, d_model, k2_heads, k2_layers, k2_ff, dropout, activation, norm_first
        )
        self.block3 = TemporalTransformer(
            d_model, k3_heads, k3_layers, k3_ff, dropout, activation, norm_first
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        # Linear: Xavier uniform · Conv: Kaiming · CLS / Embedding: trunc normal
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
        # MHA packed in_proj (Q,K,V) is not an nn.Linear -> Xavier as well
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.zeros_(m.in_proj_bias)
        nn.init.trunc_normal_(self.block2.cls_token, std=0.02)
        nn.init.trunc_normal_(self.block3.cls_token, std=0.02)

    def preprocess(self, x):
        # (B, C, T) / (B, R*A, S, T) / (B, R, A, S, T) -> (B, R, S, T)
        B, T = x.shape[0], x.shape[-1]
        x = x.reshape(B, self.n_rx, self.n_ant, self.n_sub, T)
        x = x[:, :, 0]                                      # antenna[0] of each RX
        if self.sample_norm:
            # z-score per (rx, sub) per sample, along time
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-6
            x = (x - mean) / std
        return x

    def forward(self, x):
        x = self.preprocess(x)                              # (B, R, S, T)
        x = self.block1(x)                                  # (B, R, n_win, S, D)
        x = self.block2(x)                                  # (B, n_win, D)
        x = self.block3(x)                                  # (B, D)
        return self.classifier(x)                           # (B, n_class) logits
