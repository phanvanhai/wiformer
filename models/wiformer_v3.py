"""
WiFormer v3 - see WiFormer_v3_Architecture.md

Input (from datasets.py)
------------------------
2d : (B, N_rx * N_ant * N_sub, T)     -- default loader output
3d : (B, N_rx * N_ant, N_sub, T)
4d : (B, N_rx, N_ant, N_sub, T)
Channel order is (rx, antenna, sub) row-major; antenna[0] of each RX is kept.

Pipeline (B = batch, R = views, S = subcarriers, K = n_win, D = d_model)
--------
preprocess  : x - mean_t per (rx, sub), / std per (sample, rx)        (B, R, S, T)
main branch : Block 1 Conv1d per (rx, sub, window)                     (B, R, K, S, D)
              Block 2 Rx x Sub attention per window, relative bias:
                same view -> b_sub[|ds|], other view -> c_cross, CLS untied
                                                                       h (B, K, D)
aux branch  : rfft per (rx, sub, window) -> log1p|X| -> drop DC
              -> mean over S -> shared MLP per view -> [mean_R || max_R]
                                                                       f (B, K, D)
fusion = "late" (default, option C): two streams, late fusion of logits per class
    main stream : h -> Block 3 (k3_layers)  -> classifier -> z_main
    freq stream : f -> Block 3' (k3f_layers) -> classifier -> z_freq
    z = w_c * z_main + (1 - w_c) * z_freq,  w_c = sigmoid(a_c), one a_c per class (init 0.5)
    loss = CE(z) + aux_w * [CE(z_main) + CE(z_freq)]
fusion = "gate" (first v3 design): fusion at M, then one Block 3
    x_k = h_k + g_k * P(f_k),  g_k = sigmoid(W_g h_k + b_g)
    P zero-init, modality dropout on P(f), aux head on mean_k h_k
Block 3     : temporal Transformer, signed relative bias b[i - j], CLS untied
classifier  : LayerNorm -> Dropout -> Linear

Window: W = floor(window_sec * fs) samples, made even (default 0.3 s), stride W // 2,
unless `window` (samples) is given explicitly.

Core ablation (`ablation`):
    full      - everything
    wo_freq   - main stream only   (gate: x_k = h_k)
    wo_main   - freq stream only   (gate: x_k = P(f_k))
    wo_block1 - Block 1 = Linear(W, D) on the raw window
    wo_block2 - Block 2 = mean over the R*S tokens of a window
    wo_block3 - Block 3 = mean over the windows
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .wiformer import WIFORMER_DATASET_CFG


# Sampling rate (Hz) per dataset, used to turn window_sec into samples.
# None = unknown: pass `fs` or `window` explicitly.
DATASET_FS = {
    "sshar_esp": 200,     # 1000 points = 5 s
    "sshar_asus": None,
    "xrf55": None,
    "ut_har": 125,        # 1 kHz x 2 s = 2000 points, resampled to 250 -> 125 Hz
}

ABLATIONS = ("full", "wo_freq", "wo_main", "wo_block1", "wo_block2", "wo_block3")
FUSIONS = ("late", "gate")


# ============================================================
# Transformer encoder layer with an additive attention bias
# ============================================================
class BiasEncoderLayer(nn.Module):
    """
    Pre-LN / post-LN Transformer encoder layer whose attention takes an
    additive float bias of shape (H, L, L), broadcast over the batch
    (scaled_dot_product_attention does not materialise a (B*H, L, L) copy).
    Same parameter count as nn.TransformerEncoderLayer.
    """

    def __init__(self, d_model, n_heads, dim_ff, dropout, activation="gelu", norm_first=True):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm_first = norm_first
        self.attn_drop = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, dim_ff)
        self.ff2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop_ff = nn.Dropout(dropout)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def _sa(self, x, bias):
        N, L, D = x.shape
        q, k, v = self.qkv(x).view(N, L, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        mask = None if bias is None else bias.unsqueeze(0).to(q.dtype)   # (1, H, L, L)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )                                                                  # (N, H, L, d_head)
        return self.drop1(self.out_proj(out.transpose(1, 2).reshape(N, L, D)))

    def _ff(self, x):
        return self.drop2(self.ff2(self.drop_ff(self.act(self.ff1(x)))))

    def forward(self, x, bias=None):
        if self.norm_first:
            x = x + self._sa(self.norm1(x), bias)
            x = x + self._ff(self.norm2(x))
        else:
            x = self.norm1(x + self._sa(x, bias))
            x = self.norm2(x + self._ff(x))
        return x


def _with_cls_bias(body, cls):
    """
    body : (H, L, L) bias between non-CLS tokens
    cls  : (H, 3)    [CLS->token, token->CLS, CLS->CLS]
    -> (H, L+1, L+1) with CLS at index 0, independent of position (CLS untied).
    """
    H, L, _ = body.shape
    top = torch.cat([cls[:, 2:3].unsqueeze(-1), cls[:, 0:1].unsqueeze(-1).expand(H, 1, L)], dim=2)
    rest = torch.cat([cls[:, 1:2].unsqueeze(-1).expand(H, L, 1), body], dim=2)
    return torch.cat([top, rest], dim=1)


# ============================================================
# Block 1 - Conv1d on every (rx, sub, window) segment
# ============================================================
class ConvExtractor(nn.Module):
    """
    3 x Conv1d (+ BN + GELU) -> AdaptiveAvgPool1d(n_pool) -> flatten -> Linear(feat*n_pool, D).
    raw=True (ablation wo_block1): Linear(W, D) on the raw window.
    """

    def __init__(self, d_model, feat, window, n_conv=3, pool="flat4", raw=False):
        super().__init__()
        self.raw = raw
        if raw:
            self.proj = nn.Linear(window, d_model)
            return
        if n_conv not in (2, 3, 4):
            raise ValueError(f"k1_conv must be 2, 3 or 4, got {n_conv}")
        if pool == "avg":
            n_pool = 1
        elif pool.startswith("flat") and pool[4:].isdigit():
            n_pool = int(pool[4:])
        else:
            raise ValueError(f"k1_pool must be avg or flatN, got '{pool}'")
        c1 = max(32, feat // 2)
        layers = [
            nn.Conv1d(1, c1, kernel_size=8, stride=2, padding=3), nn.BatchNorm1d(c1), nn.GELU(),
            nn.Conv1d(c1, feat, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(feat), nn.GELU(),
        ]
        for _ in range(n_conv - 2):
            layers += [nn.Conv1d(feat, feat, kernel_size=3, padding=1), nn.BatchNorm1d(feat), nn.GELU()]
        layers.append(nn.AdaptiveAvgPool1d(n_pool))
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Linear(feat * n_pool, d_model)

    def forward(self, seg):
        # seg: (N, W) -> (N, D)
        if self.raw:
            return self.proj(seg)
        return self.proj(self.encoder(seg.unsqueeze(1)).flatten(1))


# ============================================================
# Block 2 - Rx x Sub attention per window, relative bias
# ============================================================
class RxSubAttention(nn.Module):
    """
    Tokens: the R*S (view, sub) tokens of one window + CLS. No absolute PE.
    Bias (shared by all layers):
        same view  : b_sub[h, |s_i - s_j|]   (S values / head, symmetric)
        other view : c_cross[h]              (1 value / head)
        CLS        : 3 values / head (CLS->token, token->CLS, CLS->CLS)
    Only same/other-view relations are used -> equivariant to permuting the views.
    pool=True (ablation wo_block2): mean over the R*S tokens.
    """

    def __init__(self, n_view, n_sub, d_model, n_heads, n_layers, dim_ff, dropout,
                 activation="gelu", norm_first=True, pool=False):
        super().__init__()
        self.pool = pool
        if pool:
            return
        self.n_heads = n_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sub_bias = nn.Parameter(torch.zeros(n_heads, n_sub))
        self.cross_bias = nn.Parameter(torch.zeros(n_heads))
        self.cls_bias = nn.Parameter(torch.zeros(n_heads, 3))
        self.layers = nn.ModuleList([
            BiasEncoderLayer(d_model, n_heads, dim_ff, dropout, activation, norm_first)
            for _ in range(n_layers)
        ])
        view = torch.arange(n_view).repeat_interleave(n_sub)            # token -> view
        sub = torch.arange(n_sub).repeat(n_view)                         # token -> sub
        self.register_buffer("rel_sub", (sub[:, None] - sub[None, :]).abs(), persistent=False)
        self.register_buffer("same_view", view[:, None] == view[None, :], persistent=False)

    def bias(self):
        H = self.n_heads
        body = torch.where(
            self.same_view.unsqueeze(0),
            self.sub_bias[:, self.rel_sub],                              # (H, RS, RS)
            self.cross_bias.view(H, 1, 1),
        )
        return _with_cls_bias(body, self.cls_bias)                       # (H, RS+1, RS+1)

    def forward(self, x):
        # x: (B, R, K, S, D) -> h: (B, K, D)
        B, R, K, S, D = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * K, R * S, D)
        if self.pool:
            return x.mean(dim=1).reshape(B, K, D)
        x = torch.cat([self.cls_token.expand(B * K, -1, -1), x], dim=1)
        bias = self.bias()
        for layer in self.layers:
            x = layer(x, bias)
        return x[:, 0].reshape(B, K, D)


# ============================================================
# Aux branch - speed spectrum per window
# ============================================================
class SpeedSpectrum(nn.Module):
    """
    Per (view, sub, window): rfft(W) -> log1p|X| -> drop DC  (W//2 bins)
    -> mean over subcarriers (in the spectral domain: no cancellation)
    -> shared MLP per view -> [mean_R || max_R] -> Linear(2D, D)
    Symmetric over views (no view identity).
    """

    def __init__(self, window, d_model):
        super().__init__()
        self.n_bins = window // 2
        self.mlp = nn.Sequential(
            nn.Linear(self.n_bins, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.pool_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, seg):
        # seg: (B, R, S, K, W) -> f: (B, K, D)
        spec = torch.fft.rfft(seg.float(), dim=-1).abs()
        spec = torch.log1p(spec)[..., 1:1 + self.n_bins].to(seg.dtype)  # (B, R, S, K, n_bins)
        e = self.mlp(spec.mean(dim=2))                                    # (B, R, K, D)
        return self.pool_proj(torch.cat([e.mean(dim=1), e.amax(dim=1)], dim=-1))


# ============================================================
# Block 3 - temporal Transformer, signed relative bias
# ============================================================
class TemporalTransformer(nn.Module):
    """
    Window tokens + CLS. Bias (shared by all layers):
        two windows : b[h, i - j], i - j in {-(K-1), ..., K-1}  (signed, T5 / Swin style)
        CLS         : 3 values / head (untied from position)
    n_layers = 0 (ablation wo_block3): mean over the windows.
    """

    def __init__(self, n_win, d_model, n_heads, n_layers, dim_ff, dropout,
                 activation="gelu", norm_first=True):
        super().__init__()
        self.n_layers = n_layers
        if n_layers == 0:
            return
        self.n_heads = n_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.rel_bias = nn.Parameter(torch.zeros(n_heads, 2 * n_win - 1))
        self.cls_bias = nn.Parameter(torch.zeros(n_heads, 3))
        self.layers = nn.ModuleList([
            BiasEncoderLayer(d_model, n_heads, dim_ff, dropout, activation, norm_first)
            for _ in range(n_layers)
        ])
        pos = torch.arange(n_win)
        self.register_buffer("rel_idx", pos[:, None] - pos[None, :] + n_win - 1, persistent=False)

    def bias(self):
        return _with_cls_bias(self.rel_bias[:, self.rel_idx], self.cls_bias)

    def forward(self, x):
        # x: (B, K, D) -> (B, D)
        if self.n_layers == 0:
            return x.mean(dim=1)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        bias = self.bias()
        for layer in self.layers:
            x = layer(x, bias)
        return x[:, 0]


# ============================================================
# Full model
# ============================================================
class WiFormerV3(BaseModel):
    def __init__(
        self,
        input_shape,
        num_classes,
        dataset=None,
        n_rx=None,
        n_ant=None,
        n_sub=None,
        fs=None,
        window_sec=0.3,
        window=None,
        stride=None,
        d_model=64,
        feat=64,
        k1_conv=3,
        k1_pool="flat4",
        k2_layers=2,
        k2_heads=4,
        k2_ff=None,
        k3_layers=6,
        k3_heads=4,
        k3_ff=None,
        dropout=0.1,
        activation="gelu",
        norm_first=True,
        mdrop=0.3,
        aux_w=0.3,
        gate_bias=-2.0,
        ablation="full",
        fusion="late",
        k3f_layers=2,
        **kwargs,
    ):
        super().__init__(input_shape=input_shape, num_classes=num_classes)
        if ablation not in ABLATIONS:
            raise ValueError(f"WiFormerV3: ablation must be one of {ABLATIONS}, got '{ablation}'")
        if fusion not in FUSIONS:
            raise ValueError(f"WiFormerV3: fusion must be one of {FUSIONS}, got '{fusion}'")
        self.ablation = ablation
        self.fusion = fusion

        name = (dataset or "").lower()
        cfg = dict(WIFORMER_DATASET_CFG.get(name, {}))
        self.n_rx = n_rx or cfg.get("n_rx")
        self.n_ant = n_ant or cfg.get("n_ant", 1)
        self.n_sub = n_sub or cfg.get("n_sub")
        if self.n_rx is None or self.n_sub is None:
            raise ValueError(f"WiFormerV3: unknown layout for dataset '{dataset}'. Pass n_rx / n_ant / n_sub.")
        T = input_shape[-1]
        if math.prod(input_shape[:-1]) != self.n_rx * self.n_ant * self.n_sub:
            raise ValueError(
                f"WiFormerV3: input_shape {tuple(input_shape)} does not match "
                f"n_rx*n_ant*n_sub = {self.n_rx}*{self.n_ant}*{self.n_sub}"
            )

        # ---- window in samples ----
        self.fs = fs or DATASET_FS.get(name)
        if window is None or window <= 0:
            if self.fs is None:
                raise ValueError(
                    f"WiFormerV3: sampling rate of '{dataset}' is unknown. "
                    "Pass --fs (Hz) or --window (samples)."
                )
            window = int(window_sec * self.fs)
            window -= window % 2
        if window < 16 or window > T:
            raise ValueError(f"WiFormerV3: window={window} samples is out of range (16..T={T})")
        stride = stride or window // 2
        self.window, self.stride = window, stride
        self.n_win = (T - window) // stride + 1
        dur = f"{window / self.fs:.3f} s" if self.fs else "fs unknown"
        print(f"[WiFormerV3] window W={window} ({dur}), stride={stride}, n_win={self.n_win}, "
              f"fusion={fusion}, ablation={ablation}")

        k2_ff = k2_ff or 4 * d_model
        k3_ff = k3_ff or 4 * d_model
        R, S = self.n_rx, self.n_sub
        self.use_main = ablation != "wo_main"
        self.use_freq = ablation != "wo_freq"
        self.two_streams = self.use_main and self.use_freq
        late = fusion == "late"
        self.use_gate = self.two_streams and not late
        # late: deep supervision on both streams; gate: aux head on the main branch
        self.aux_w = aux_w if (self.two_streams if late else self.use_main) else 0.0
        self.mdrop = mdrop
        n_layers3 = 0 if ablation == "wo_block3" else k3_layers
        n_layers3f = 0 if ablation == "wo_block3" else k3f_layers

        # ---- main branch ----
        if self.use_main:
            self.block1 = ConvExtractor(d_model, feat, window, k1_conv, k1_pool, raw=ablation == "wo_block1")
            self.block2 = RxSubAttention(
                R, S, d_model, k2_heads, k2_layers, k2_ff, dropout, activation, norm_first,
                pool=ablation == "wo_block2",
            )
        if self.use_freq:
            self.speed = SpeedSpectrum(window, d_model)

        def head():
            return nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

        if late:
            # ---- option C: one Block 3 + classifier per stream, per-class late fusion ----
            if self.use_main:
                self.block3 = TemporalTransformer(self.n_win, d_model, k3_heads, n_layers3,
                                                  k3_ff, dropout, activation, norm_first)
                self.classifier = head()
            if self.use_freq:
                self.block3f = TemporalTransformer(self.n_win, d_model, k3_heads, n_layers3f,
                                                   k3_ff, dropout, activation, norm_first)
                self.classifier_f = head()
            if self.two_streams:
                self.branch_logit = nn.Parameter(torch.zeros(num_classes))   # w_c = sigmoid(a_c) = 0.5
        else:
            # ---- first v3 design: gated residual at M, one Block 3 ----
            if self.use_freq:
                self.proj_f = nn.Linear(d_model, d_model)                     # P
            if self.use_gate:
                self.gate = nn.Linear(d_model, d_model)                       # W_g, b_g
            self.block3 = TemporalTransformer(self.n_win, d_model, k3_heads, n_layers3,
                                              k3_ff, dropout, activation, norm_first)
            self.classifier = head()
            if self.aux_w > 0:
                self.aux_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

        self._aux_logits = None        # gate: aux head logits (training)
        self._stream_logits = None     # late: (z_main, z_freq)
        self._last_gate = None
        self._init_weights(gate_bias)

    # --------------------------------------------------------
    def _init_weights(self, gate_bias):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for blk in (getattr(self, "block2", None), getattr(self, "block3", None), getattr(self, "block3f", None)):
            if blk is not None and hasattr(blk, "cls_token"):
                nn.init.trunc_normal_(blk.cls_token, std=0.02)
        if self.use_gate:
            # start from the time-only solution: P(f) = 0, gate mostly closed
            nn.init.zeros_(self.proj_f.weight)
            nn.init.zeros_(self.proj_f.bias)
            nn.init.constant_(self.gate.bias, gate_bias)

    # --------------------------------------------------------
    def preprocess(self, x):
        # (B, C, T) / (B, R*A, S, T) / (B, R, A, S, T) -> (B, R, S, T)
        B, T = x.shape[0], x.shape[-1]
        x = x.reshape(B, self.n_rx, self.n_ant, self.n_sub, T)[:, :, 0]
        x = x - x.mean(dim=-1, keepdim=True)                              # static level per (rx, sub)
        std = x.flatten(2).std(dim=-1).view(B, self.n_rx, 1, 1) + 1e-6    # one std per (sample, rx)
        return x / std

    def forward(self, x):
        x = self.preprocess(x)                                            # (B, R, S, T)
        B, R, S, _ = x.shape
        seg = x.unfold(-1, self.window, self.stride)                      # (B, R, S, K, W)
        K = seg.shape[3]

        h = None
        if self.use_main:
            v = self.block1(seg.permute(0, 1, 3, 2, 4).reshape(-1, self.window))
            h = self.block2(v.reshape(B, R, K, S, -1))                     # (B, K, D)

        if self.fusion == "late":
            z_main = self.classifier(self.block3(h)) if self.use_main else None
            z_freq = self.classifier_f(self.block3f(self.speed(seg))) if self.use_freq else None
            if not self.two_streams:
                self._stream_logits = None
                return z_main if z_main is not None else z_freq
            self._stream_logits = (z_main, z_freq)
            w = torch.sigmoid(self.branch_logit)                           # (C,) weight of the main stream
            return w * z_main + (1.0 - w) * z_freq

        if self.use_freq:
            pf = self.proj_f(self.speed(seg))                              # (B, K, D)
            if self.use_gate:
                g = torch.sigmoid(self.gate(h))
                self._last_gate = g.detach()
                u = g * pf
                if self.training and self.mdrop > 0:
                    keep = (torch.rand(B, 1, 1, device=u.device) >= self.mdrop).to(u.dtype)
                    u = u * keep / (1.0 - self.mdrop)
                tokens = h + u
            else:
                tokens = pf                                                # wo_main
        else:
            tokens = h                                                     # wo_freq

        if self.aux_w > 0 and self.training:
            self._aux_logits = self.aux_head(h.mean(dim=1))
        else:
            self._aux_logits = None

        return self.classifier(self.block3(tokens))

    def extra_loss(self, y):
        """
        Auxiliary loss (training only), called by train.py after the main CE:
        late : aux_w * [CE(z_main) + CE(z_freq)]  (each stream must classify on its own)
        gate : aux_w * CE(aux head on the main branch)
        """
        if not self.training or self.aux_w <= 0:
            return None
        if self.fusion == "late":
            if self._stream_logits is None:
                return None
            z_main, z_freq = self._stream_logits
            return self.aux_w * (F.cross_entropy(z_main, y) + F.cross_entropy(z_freq, y))
        if self._aux_logits is None:
            return None
        return self.aux_w * F.cross_entropy(self._aux_logits, y)

    # --------------------------------------------------------
    @torch.no_grad()
    def analyze(self, loader, device, save_dir):
        """
        After training (best checkpoint), on the test loader:
        - stream_by_class.csv (late, two streams): per true class, the main-stream
          weight w_c and the recall of z_main / z_freq / fused z
        - gate_by_class.csv  (gate, full model): mean gate g per true class
        - block2_bias.json / block3_bias.json / block3f_bias.json : learned bias tables
        - rx_permutation.json: prediction agreement when the views are permuted
        """
        self.eval()
        os.makedirs(save_dir, exist_ok=True)
        R = self.n_rx
        perm = list(range(1, R)) + [0]                                    # cyclic shift of the views
        gate_sum, gate_cnt = {}, {}
        hit = {}                                                          # class -> [n, main, freq, fused]
        n, agree, max_diff = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = self(x)
            if self._stream_logits is not None:
                z_main, z_freq = self._stream_logits
                for c in y.unique().tolist():
                    m = y == c
                    s = hit.setdefault(c, [0, 0, 0, 0])
                    s[0] += m.sum().item()
                    s[1] += (z_main[m].argmax(1) == c).sum().item()
                    s[2] += (z_freq[m].argmax(1) == c).sum().item()
                    s[3] += (logits[m].argmax(1) == c).sum().item()
            if self._last_gate is not None and self.use_gate:
                g = self._last_gate.mean(dim=(1, 2))
                for c in y.unique().tolist():
                    sel = g[y == c]
                    gate_sum[c] = gate_sum.get(c, 0.0) + sel.sum().item()
                    gate_cnt[c] = gate_cnt.get(c, 0) + sel.numel()
            xp = x.reshape(x.shape[0], R, -1, x.shape[-1])[:, perm].reshape(x.shape)
            logits_p = self(xp)
            agree += (logits.argmax(1) == logits_p.argmax(1)).sum().item()
            max_diff = max(max_diff, (logits - logits_p).abs().max().item())
            n += y.numel()

        if hit:
            w = torch.sigmoid(self.branch_logit).tolist()
            with open(os.path.join(save_dir, "stream_by_class.csv"), "w", encoding="utf-8") as fh:
                fh.write("class,w_main,w_freq,recall_main,recall_freq,recall_fused,n\n")
                for c in sorted(hit):
                    k, a, b, f = hit[c]
                    fh.write(f"{c},{w[c]:.4f},{1 - w[c]:.4f},{a / k:.4f},{b / k:.4f},{f / k:.4f},{k}\n")
                tot = [sum(v[i] for v in hit.values()) for i in range(4)]
                fh.write(f"all,,,{tot[1] / tot[0]:.4f},{tot[2] / tot[0]:.4f},{tot[3] / tot[0]:.4f},{tot[0]}\n")
            print(f"[WiFormerV3] test acc  main stream {tot[1] / tot[0]:.4f} | "
                  f"freq stream {tot[2] / tot[0]:.4f} | fused {tot[3] / tot[0]:.4f}")
        if gate_cnt:
            with open(os.path.join(save_dir, "gate_by_class.csv"), "w", encoding="utf-8") as fh:
                fh.write("class,mean_gate,n\n")
                for c in sorted(gate_cnt):
                    fh.write(f"{c},{gate_sum[c] / gate_cnt[c]:.6f},{gate_cnt[c]}\n")
        with open(os.path.join(save_dir, "rx_permutation.json"), "w", encoding="utf-8") as fh:
            json.dump({"perm": perm, "n": n, "agreement": agree / max(n, 1), "max_abs_logit_diff": max_diff},
                      fh, indent=4)
        if self.use_main and not self.block2.pool:
            b2 = self.block2
            with open(os.path.join(save_dir, "block2_bias.json"), "w", encoding="utf-8") as fh:
                json.dump({"sub_bias[h][|ds|]": b2.sub_bias.tolist(),
                           "cross_bias[h]": b2.cross_bias.tolist(),
                           "cls_bias[h][cls->tok, tok->cls, cls->cls]": b2.cls_bias.tolist()}, fh, indent=2)
        for attr, fname in (("block3", "block3_bias.json"), ("block3f", "block3f_bias.json")):
            b3 = getattr(self, attr, None)
            if b3 is None or b3.n_layers == 0:
                continue
            with open(os.path.join(save_dir, fname), "w", encoding="utf-8") as fh:
                json.dump({"offsets(i-j)": list(range(-(self.n_win - 1), self.n_win)),
                           "rel_bias[h][offset]": b3.rel_bias.tolist(),
                           "cls_bias[h][cls->win, win->cls, cls->cls]": b3.cls_bias.tolist()}, fh, indent=2)
        print(f"[WiFormerV3] RX permutation agreement = {agree / max(n, 1):.4f} "
              f"(max |logit diff| = {max_diff:.2e})")
