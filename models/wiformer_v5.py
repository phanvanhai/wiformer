"""
WiFormer v5 - an action is a sequence of short moments.

Idea
----
The CSI is cut into short overlapping moments (windows, ~0.3 s). Inside one moment:
  - shape of the motion : seen differently by every channel (rx, subcarrier)
                          -> look at every channel, then aggregate selectively (attention)
  - speed of the motion : the same for every channel (Doppler of the body)
                          -> average the spectrum over all channels (Hann window, STFT style)
Across moments: read the moments in order (signed relative bias).
The shape branch is far larger than the speed branch, so during training it is dropped
for a whole sample with probability `mod_drop` (modality dropout): the temporal block must
also learn to read the speed branch.

Architecture = WiFormer v4 ablation `hybrid` + `--mod_drop=0.5` (same layers and parameters)
               + Hann window before the FFT of the speed branch.

Input (from datasets.py)
------------------------
2d : (B, N_rx * N_ant * N_sub, T) ; 3d : (B, N_rx * N_ant, N_sub, T) ; 4d : (B, N_rx, N_ant, N_sub, T)
Channel order is (rx, antenna, sub) row-major; antenna[0] of each RX is kept.

Pipeline (R = rx, S = subcarriers, K = n_win, W = window, D = d_model)
--------
preprocess : norm="sub"  -> x - mean_t per (rx, sub), / std per (sample, rx)
             norm="none" -> input as given by the loader              (B, R, S, T)
windows    : W = 2^round(log2(window_sec * fs)), stride W // 2       (B, R, S, K, W)
shape      : Block 1  token = Linear(D, D)(Conv1d x3 -> pool flat4 -> Linear)   per (rx, sub, window)
             Block 2  Rx x Sub attention per window, relative bias, CLS -> h_k  (B, K, D)
speed      : per-window mean removed -> Hann (scaled to mean 1) -> log1p|rfft| without DC (W//2 bins),
             mean over (rx, sub) -> Linear -> a_k                     (B, K, D)
fusion     : z_k = Linear(2D, D)([h_k || a_k]); training: h = 0 for a sample w.p. mod_drop
Block 3    : temporal Transformer, signed relative bias b[i - j], CLS
classifier : LayerNorm -> Dropout -> Linear                           (B, C);  loss = CE

Ablations: wo_shape = speed branch only, z_k = Linear(D, D)(a_k) -> Block 3
           spec_win="rect" (ablation rect) = no window before the FFT (= v4 hybrid_drop).
"""

import json
import math
import os

import torch
import torch.nn as nn

from .base import BaseModel
from .wiformer import WIFORMER_DATASET_CFG
from .wiformer_v4 import DATASET_FS, ConvExtractor, RxSubAttention, TemporalTransformer

ABLATIONS = ("full", "wo_shape")
SPEC_WINDOWS = ("rect", "hann")


class SpeedBranch(nn.Module):
    """
    log1p|rfft(W)| without DC (W//2 bins), mean over the R*S channels -> Linear(W//2, D).
    spec_win="rect": no window (ablation).
    spec_win="hann" (default): per-window mean removed, then periodic Hann window scaled to mean 1
    (same gain as rect) before the FFT -> sidelobes -31 dB instead of -13 dB; Hann with the
    50 % overlap sums to a constant (COLA), so no part of the signal is down-weighted overall.
    """

    def __init__(self, window, d_model, spec_win="hann"):
        super().__init__()
        self.n_bins = window // 2
        self.spec_win = spec_win
        if spec_win == "hann":
            w = torch.hann_window(window, periodic=True)
            self.register_buffer("win", w / w.mean(), persistent=False)
        self.proj = nn.Linear(self.n_bins, d_model)

    def forward(self, seg):
        # seg: (B, R, S, K, W) -> (B, K, D)
        s = seg.float()
        if self.spec_win == "hann":
            s = (s - s.mean(dim=-1, keepdim=True)) * self.win
        spec = torch.log1p(torch.fft.rfft(s, dim=-1).abs())[..., 1:1 + self.n_bins]
        return self.proj(spec.mean(dim=(1, 2)).to(seg.dtype))


class MomentFusion(nn.Module):
    """
    z_k = Linear(2D, D)([h_k || a_k]).
    mod_drop : training only, zero h (whole sequence of a sample) with probability mod_drop.
    `knock` ("shape" / "speed" / None) zeroes one branch at inference (analysis only).
    """

    def __init__(self, d_model, mod_drop=0.5):
        super().__init__()
        self.fuse = nn.Linear(2 * d_model, d_model)
        self.mod_drop = mod_drop
        self.knock = None

    def forward(self, h, a):
        if self.training and self.mod_drop > 0:
            keep = torch.rand(h.shape[0], 1, 1, device=h.device) >= self.mod_drop
            h = h * keep.to(h.dtype)
        if self.knock == "shape":
            h = torch.zeros_like(h)
        elif self.knock == "speed":
            a = torch.zeros_like(a)
        return self.fuse(torch.cat([h, a], dim=-1))


class WiFormerV5(BaseModel):
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
        mod_drop=0.5,
        spec_win="hann",
        **kwargs,
    ):
        super().__init__(input_shape=input_shape, num_classes=num_classes)
        if ablation not in ABLATIONS:
            raise ValueError(f"WiFormerV5: ablation must be one of {ABLATIONS}, got '{ablation}'")
        if norm not in ("sub", "none"):
            raise ValueError(f"WiFormerV5: norm must be 'sub' or 'none', got '{norm}'")
        if spec_win not in SPEC_WINDOWS:
            raise ValueError(f"WiFormerV5: spec_win must be one of {SPEC_WINDOWS}, got '{spec_win}'")
        self.ablation, self.norm = ablation, norm
        if ablation == "wo_shape":
            mod_drop = 0.0                                                # no shape branch to drop

        name = (dataset or "").lower()
        cfg = dict(WIFORMER_DATASET_CFG.get(name, {}))
        self.n_rx = n_rx or cfg.get("n_rx")
        self.n_ant = n_ant or cfg.get("n_ant", 1)
        self.n_sub = n_sub or cfg.get("n_sub")
        if self.n_rx is None or self.n_sub is None:
            raise ValueError(f"WiFormerV5: unknown layout for dataset '{dataset}'. Pass n_rx / n_ant / n_sub.")
        T = input_shape[-1]
        if math.prod(input_shape[:-1]) != self.n_rx * self.n_ant * self.n_sub:
            raise ValueError(f"WiFormerV5: input_shape {tuple(input_shape)} does not match "
                             f"n_rx*n_ant*n_sub = {self.n_rx}*{self.n_ant}*{self.n_sub}")

        self.fs = fs or DATASET_FS.get(name)
        if window is None or window <= 0:
            if self.fs is None:
                raise ValueError(f"WiFormerV5: sampling rate of '{dataset}' is unknown. Pass --fs or --window.")
            window = 2 ** round(math.log2(window_sec * self.fs))
        if window < 16 or window > T:
            raise ValueError(f"WiFormerV5: window={window} samples is out of range (16..T={T})")
        stride = stride or window // 2
        self.window, self.stride = window, stride
        self.n_win = (T - window) // stride + 1
        dur = f"{window / self.fs:.3f} s" if self.fs else "fs unknown"
        print(f"[WiFormerV5] window W={window} ({dur}), stride={stride}, n_win={self.n_win}, "
              f"norm={norm}, mod_drop={mod_drop}, spec_win={spec_win}, ablation={ablation}")

        k2_ff = k2_ff or 4 * d_model
        k3_ff = k3_ff or 4 * d_model
        R, S = self.n_rx, self.n_sub

        if ablation != "wo_shape":
            # shape: per-channel waveform tokens -> selective aggregation over channels
            self.conv = ConvExtractor(d_model, feat, window, k1_conv, k1_pool)
            self.token = nn.Linear(d_model, d_model)
            self.block2 = RxSubAttention(R, S, d_model, k2_heads, k2_layers, k2_ff, dropout,
                                         activation, norm_first)
        # speed: spectrum averaged over all channels
        self.speed = SpeedBranch(window, d_model, spec_win)
        # wo_shape (speed only): z_k = Linear(D, D)(a_k)  (same function class as v4 freq_mean)
        self.fusion = nn.Linear(d_model, d_model) if ablation == "wo_shape" else MomentFusion(d_model, mod_drop)
        # order: temporal Transformer over the moments
        self.block3 = TemporalTransformer(self.n_win, d_model, k3_heads, k3_layers, k3_ff, dropout,
                                          activation, norm_first)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout),
                                        nn.Linear(d_model, num_classes))
        self._init_weights()

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
        for blk in (getattr(self, "block2", None), self.block3):
            if blk is not None:
                nn.init.trunc_normal_(blk.cls_token, std=0.02)

    def preprocess(self, x):
        B, T = x.shape[0], x.shape[-1]
        x = x.reshape(B, self.n_rx, self.n_ant, self.n_sub, T)[:, :, 0]
        if self.norm == "none":
            return x
        x = x - x.mean(dim=-1, keepdim=True)                              # static level per (rx, sub)
        std = x.flatten(2).std(dim=-1).view(B, self.n_rx, 1, 1) + 1e-6    # one std per (sample, rx)
        return x / std

    def forward(self, x):
        x = self.preprocess(x)                                            # (B, R, S, T)
        B, R, S, _ = x.shape
        seg = x.unfold(-1, self.window, self.stride)                      # (B, R, S, K, W)
        K = seg.shape[3]
        if self.ablation == "wo_shape":
            return self.classifier(self.block3(self.fusion(self.speed(seg))))
        v = self.token(self.conv(seg.permute(0, 1, 3, 2, 4).reshape(-1, self.window)))
        h = self.block2(v.reshape(B, R, K, S, -1))                        # shape  (B, K, D)
        a = self.speed(seg)                                               # speed  (B, K, D)
        return self.classifier(self.block3(self.fusion(h, a)))

    @torch.no_grad()
    def analyze(self, loader, device, save_dir):
        """
        After training (best checkpoint), on the test loader:
        - view_contribution.json : fuse weight norm per branch and test accuracy (overall +
          per class) when one branch is zeroed (knock-out: shape / speed)
        - block2_bias.json / block3_bias.json : learned bias tables
        - rx_permutation.json : prediction agreement when the RX are cyclically permuted
        """
        self.eval()
        os.makedirs(save_dir, exist_ok=True)
        R = self.n_rx
        perm = list(range(1, R)) + [0]
        b3 = self.block3
        with open(os.path.join(save_dir, "block3_bias.json"), "w", encoding="utf-8") as fh:
            json.dump({"offsets(i-j)": list(range(-(self.n_win - 1), self.n_win)),
                       "rel_bias[h][offset]": b3.rel_bias.tolist(),
                       "cls_bias[h][cls->win, win->cls, cls->cls]": b3.cls_bias.tolist()}, fh, indent=2)
        if self.ablation == "wo_shape":
            return                                                        # one branch: no knock-out
        fu = self.fusion
        modes = [None, "shape", "speed"]
        hit = {m: {} for m in modes}
        n, agree, max_diff = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            for mode in modes:
                fu.knock = mode
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
        fu.knock = None

        def acc(t):
            return sum(v[1] for v in t.values()) / max(sum(v[0] for v in t.values()), 1)

        D = fu.fuse.weight.shape[0]
        w = fu.fuse.weight
        names = {None: "both", "shape": "speed_only(shape zeroed)", "speed": "shape_only(speed zeroed)"}
        out = {
            "fuse_weight_norm": {"shape": w[:, :D].norm().item(), "speed": w[:, D:].norm().item()},
            "test_acc": {names[m]: acc(hit[m]) for m in modes},
            "recall_by_class": {names[m]: {str(c): hit[m][c][1] / hit[m][c][0] for c in sorted(hit[m])}
                                for m in modes},
        }
        with open(os.path.join(save_dir, "view_contribution.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        with open(os.path.join(save_dir, "rx_permutation.json"), "w", encoding="utf-8") as fh:
            json.dump({"perm": perm, "n": n, "agreement": agree / max(n, 1), "max_abs_logit_diff": max_diff},
                      fh, indent=4)
        b2 = self.block2
        with open(os.path.join(save_dir, "block2_bias.json"), "w", encoding="utf-8") as fh:
            json.dump({"sub_bias[h][|ds|]": b2.sub_bias.tolist(),
                       "cross_bias[h]": b2.cross_bias.tolist(),
                       "cls_bias[h][cls->tok, tok->cls, cls->cls]": b2.cls_bias.tolist()}, fh, indent=2)
        print("[WiFormerV5] test acc both {:.4f} | speed only {:.4f} | shape only {:.4f} | "
              "RX permutation agreement {:.4f}".format(acc(hit[None]), acc(hit["shape"]), acc(hit["speed"]),
                                                       agree / max(n, 1)))
