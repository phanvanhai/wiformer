"""
WiFormer v4 - see WiFormer_v4_Architecture.md

Input (from datasets.py)
------------------------
2d : (B, N_rx * N_ant * N_sub, T)     -- default loader output
3d : (B, N_rx * N_ant, N_sub, T)
4d : (B, N_rx, N_ant, N_sub, T)
Channel order is (rx, antenna, sub) row-major; antenna[0] of each RX is kept.

Pipeline (B = batch, R = views, S = subcarriers, K = n_win, W = window, D = d_model)
--------
preprocess : norm="sub"  -> x - mean_t per (rx, sub), / std per (sample, rx)
             norm="none" -> input as given by the loader              (B, R, S, T)
windows    : W = 2^round(log2(window_sec * fs)), stride W // 2       (B, R, S, K, W)
             (SSHAR-ESP 64 = 0.32 s, UT_HAR 32 = 0.256 s)
Block 1    : per (rx, sub, window) segment, shared weights
               v_time = Linear(Conv1d x3 -> pool flat4)               (D)
               v_freq = Linear(log1p|rfft| without DC, W//2 bins)     (D)
               token  = Linear(2D -> D)([v_time || v_freq])           (B, R, K, S, D)
Block 2    : Rx x Sub attention per window, relative bias
               same view -> b_sub[|ds|], other view -> c_cross, CLS untied
                                                                      (B, K, D)
Block 3    : temporal Transformer, signed relative bias b[i - j], CLS untied
classifier : LayerNorm -> Dropout -> Linear                           (B, C)
loss       : CE only

Core ablation (`ablation`):
    full      - everything
    wo_fft    - Block 1 = time view only   (token = Linear(D, D)(v_time))
    wo_conv   - Block 1 = spectral view only (token = Linear(D, D)(v_freq))
    wo_block1 - Block 1 = Linear(W, D) on the raw segment
    wo_block2 - Block 2 = mean over the R*S tokens of a window
    wo_block3 - Block 3 = mean over the windows
    freq_mean - wo_conv + wo_block2: spectral view only, mean over the R*S tokens of a window.
                Token and fuse are linear, so this equals Linear(mean log-spectrum over rx, sub):
                tests "attention over per-sub spectra" (wo_conv) vs "averaged spectrum"
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


ABLATIONS = ("full", "wo_fft", "wo_conv", "wo_block1", "wo_block2", "wo_block3", "freq_mean")


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
# Block 1 - two views of every (rx, sub, window) segment, early fusion
# ============================================================
class TwoViewExtractor(nn.Module):
    """
    time     : Conv1d x n_conv (+ BN + GELU) -> AdaptiveAvgPool1d(n_pool) -> Linear -> v_time
    spectral : rfft(W) -> log1p|X| -> drop DC (W//2 bins) -> Linear(W//2, D)   -> v_freq
    fuse     : Linear(n_view * D, D) on [v_time || v_freq]
    raw=True (wo_block1): Linear(W, D) on the raw segment.
    `knock` ("time" / "freq" / None) zeroes one view at inference (analysis only).
    """

    def __init__(self, d_model, feat, window, n_conv=3, pool="flat4",
                 use_time=True, use_freq=True, raw=False):
        super().__init__()
        self.raw = raw
        self.use_time = use_time and not raw
        self.use_freq = use_freq and not raw
        self.n_bins = window // 2
        self.knock = None
        if raw:
            self.proj_raw = nn.Linear(window, d_model)
            return
        if not (self.use_time or self.use_freq):
            raise ValueError("TwoViewExtractor: at least one view is needed")
        if self.use_time:
            self.time = ConvExtractor(d_model, feat, window, n_conv, pool)
        if self.use_freq:
            self.proj_freq = nn.Linear(self.n_bins, d_model)
        self.fuse = nn.Linear((int(self.use_time) + int(self.use_freq)) * d_model, d_model)

    def views(self, seg):
        # seg: (N, W) -> list of (N, D) in the order [time, freq]
        out = []
        if self.use_time:
            v = self.time(seg)
            out.append(torch.zeros_like(v) if self.knock == "time" else v)
        if self.use_freq:
            spec = torch.fft.rfft(seg.float(), dim=-1).abs()
            spec = torch.log1p(spec)[:, 1:1 + self.n_bins].to(seg.dtype)
            v = self.proj_freq(spec)
            out.append(torch.zeros_like(v) if self.knock == "freq" else v)
        return out

    def forward(self, seg):
        if self.raw:
            return self.proj_raw(seg)
        return self.fuse(torch.cat(self.views(seg), dim=-1))


# ============================================================
# Full model
# ============================================================
class WiFormerV4(BaseModel):
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
        ablation="full",
        norm="sub",
        **kwargs,
    ):
        super().__init__(input_shape=input_shape, num_classes=num_classes)
        if ablation not in ABLATIONS:
            raise ValueError(f"WiFormerV4: ablation must be one of {ABLATIONS}, got '{ablation}'")
        if norm not in ("sub", "none"):
            raise ValueError(f"WiFormerV4: norm must be 'sub' or 'none', got '{norm}'")
        self.ablation = ablation
        self.norm = norm

        name = (dataset or "").lower()
        cfg = dict(WIFORMER_DATASET_CFG.get(name, {}))
        self.n_rx = n_rx or cfg.get("n_rx")
        self.n_ant = n_ant or cfg.get("n_ant", 1)
        self.n_sub = n_sub or cfg.get("n_sub")
        if self.n_rx is None or self.n_sub is None:
            raise ValueError(f"WiFormerV4: unknown layout for dataset '{dataset}'. Pass n_rx / n_ant / n_sub.")
        T = input_shape[-1]
        if math.prod(input_shape[:-1]) != self.n_rx * self.n_ant * self.n_sub:
            raise ValueError(
                f"WiFormerV4: input_shape {tuple(input_shape)} does not match "
                f"n_rx*n_ant*n_sub = {self.n_rx}*{self.n_ant}*{self.n_sub}"
            )

        # ---- window in samples ----
        self.fs = fs or DATASET_FS.get(name)
        if window is None or window <= 0:
            if self.fs is None:
                raise ValueError(
                    f"WiFormerV4: sampling rate of '{dataset}' is unknown. "
                    "Pass --fs (Hz) or --window (samples)."
                )
            # nearest power of two to window_sec * fs (ESP 60 -> 64, UT_HAR 37.5 -> 32), so the
            # Conv stack (stride 2, 2) and the flat4 pool split the window into equal steps
            window = 2 ** round(math.log2(window_sec * self.fs))
        if window < 16 or window > T:
            raise ValueError(f"WiFormerV4: window={window} samples is out of range (16..T={T})")
        stride = stride or window // 2
        self.window, self.stride = window, stride
        self.n_win = (T - window) // stride + 1
        dur = f"{window / self.fs:.3f} s" if self.fs else "fs unknown"
        print(f"[WiFormerV4] window W={window} ({dur}), stride={stride}, n_win={self.n_win}, "
              f"norm={norm}, ablation={ablation}")

        k2_ff = k2_ff or 4 * d_model
        k3_ff = k3_ff or 4 * d_model
        R, S = self.n_rx, self.n_sub

        self.block1 = TwoViewExtractor(
            d_model, feat, window, k1_conv, k1_pool,
            use_time=ablation not in ("wo_conv", "freq_mean"), use_freq=ablation != "wo_fft",
            raw=ablation == "wo_block1",
        )
        self.block2 = RxSubAttention(
            R, S, d_model, k2_heads, k2_layers, k2_ff, dropout, activation, norm_first,
            pool=ablation in ("wo_block2", "freq_mean"),
        )
        self.block3 = TemporalTransformer(
            self.n_win, d_model, k3_heads, 0 if ablation == "wo_block3" else k3_layers,
            k3_ff, dropout, activation, norm_first,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes),
        )
        self._init_weights()

    # --------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for blk in (self.block2, self.block3):
            if hasattr(blk, "cls_token"):
                nn.init.trunc_normal_(blk.cls_token, std=0.02)

    # --------------------------------------------------------
    def preprocess(self, x):
        # (B, C, T) / (B, R*A, S, T) / (B, R, A, S, T) -> (B, R, S, T)
        B, T = x.shape[0], x.shape[-1]
        x = x.reshape(B, self.n_rx, self.n_ant, self.n_sub, T)[:, :, 0]
        if self.norm == "none":
            # keep the input as given by the loader (UT_HAR: per-sample min-max),
            # i.e. keep the static level of every (rx, sub) channel
            return x
        x = x - x.mean(dim=-1, keepdim=True)                              # static level per (rx, sub)
        std = x.flatten(2).std(dim=-1).view(B, self.n_rx, 1, 1) + 1e-6    # one std per (sample, rx)
        return x / std

    def forward(self, x):
        x = self.preprocess(x)                                            # (B, R, S, T)
        B, R, S, _ = x.shape
        seg = x.unfold(-1, self.window, self.stride)                      # (B, R, S, K, W)
        K = seg.shape[3]
        v = self.block1(seg.permute(0, 1, 3, 2, 4).reshape(-1, self.window))
        h = self.block2(v.reshape(B, R, K, S, -1))                        # (B, K, D)
        return self.classifier(self.block3(h))

    # --------------------------------------------------------
    @torch.no_grad()
    def analyze(self, loader, device, save_dir):
        """
        After training (best checkpoint), on the test loader:
        - view_contribution.json : Block 1 fuse weight norm per view, and the test
          accuracy (overall + per class) when one view is zeroed (knock-out)
        - block2_bias.json / block3_bias.json : learned bias tables
        - rx_permutation.json : prediction agreement when the views are permuted
        """
        self.eval()
        os.makedirs(save_dir, exist_ok=True)
        R = self.n_rx
        perm = list(range(1, R)) + [0]                                    # cyclic shift of the views
        b1 = self.block1
        two_views = not b1.raw and b1.use_time and b1.use_freq
        modes = [None, "time", "freq"] if two_views else [None]
        hit = {m: {} for m in modes}                                      # mode -> class -> [n, correct]
        n, agree, max_diff = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            for mode in modes:
                b1.knock = mode
                logits = self(x)
                pred = logits.argmax(1)
                for c in y.unique().tolist():
                    m = y == c
                    s = hit[mode].setdefault(c, [0, 0])
                    s[0] += m.sum().item()
                    s[1] += (pred[m] == c).sum().item()
                if mode is None:
                    xp = x.reshape(x.shape[0], R, -1, x.shape[-1])[:, perm].reshape(x.shape)
                    logits_p = self(xp)
                    agree += (pred == logits_p.argmax(1)).sum().item()
                    max_diff = max(max_diff, (logits - logits_p).abs().max().item())
            n += y.numel()
        b1.knock = None

        if two_views:
            D = b1.fuse.weight.shape[0]
            w = b1.fuse.weight
            names = {None: "both", "time": "freq_only(time zeroed)", "freq": "time_only(freq zeroed)"}

            def acc(t):
                return sum(v[1] for v in t.values()) / max(sum(v[0] for v in t.values()), 1)

            out = {
                "fuse_weight_norm": {"time": w[:, :D].norm().item(), "freq": w[:, D:].norm().item()},
                "test_acc": {names[m]: acc(hit[m]) for m in modes},
                "recall_by_class": {names[m]: {str(c): hit[m][c][1] / hit[m][c][0] for c in sorted(hit[m])}
                                    for m in modes},
            }
            with open(os.path.join(save_dir, "view_contribution.json"), "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
            print("[WiFormerV4] test acc  both {:.4f} | time zeroed {:.4f} | freq zeroed {:.4f}".format(
                acc(hit[None]), acc(hit["time"]), acc(hit["freq"])))

        with open(os.path.join(save_dir, "rx_permutation.json"), "w", encoding="utf-8") as fh:
            json.dump({"perm": perm, "n": n, "agreement": agree / max(n, 1), "max_abs_logit_diff": max_diff},
                      fh, indent=4)
        if not self.block2.pool:
            b2 = self.block2
            with open(os.path.join(save_dir, "block2_bias.json"), "w", encoding="utf-8") as fh:
                json.dump({"sub_bias[h][|ds|]": b2.sub_bias.tolist(),
                           "cross_bias[h]": b2.cross_bias.tolist(),
                           "cls_bias[h][cls->tok, tok->cls, cls->cls]": b2.cls_bias.tolist()}, fh, indent=2)
        if self.block3.n_layers > 0:
            b3 = self.block3
            with open(os.path.join(save_dir, "block3_bias.json"), "w", encoding="utf-8") as fh:
                json.dump({"offsets(i-j)": list(range(-(self.n_win - 1), self.n_win)),
                           "rel_bias[h][offset]": b3.rel_bias.tolist(),
                           "cls_bias[h][cls->win, win->cls, cls->cls]": b3.cls_bias.tolist()}, fh, indent=2)
        print(f"[WiFormerV4] RX permutation agreement = {agree / max(n, 1):.4f} "
              f"(max |logit diff| = {max_diff:.2e})")
