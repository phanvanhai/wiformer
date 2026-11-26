"""
Summarise cross-user k-fold results (SSHAR, 7 classes) of several variants.

Finds every run folder named  <dataset>_user_c<fold>_<variant>  under --root (any depth),
reads <run>/test/predictions.csv and reports:
  - accuracy per fold and mean +- std (sample std) over folds:
      6 actions : test samples of the 6 action classes (a prediction 'static' counts as wrong)
      7 classes : all test samples
  - pooled McNemar (exact binomial, two-sided) for every pair of variants, on the predictions
    of all folds together (every subject is tested once), only folds present for both variants
  - pair errors summed over folds (sit<->stand, fall<->sit, walk<->run, pick<->stand,
    action -> static)
  - fusion=gate runs: g (share of the shape branch) per class and moment, averaged over folds
    -> gate_by_class_moment.csv + gate_by_class_moment.png

Usage:
  python summarize_folds.py --root results/v5 --variants full gate wo_shape [--out results/v5/summary]
"""

import argparse
import csv
import json
import math
import os
import re
from itertools import combinations

import numpy as np

CLASSES = ["sit_down", "stand_up", "pick_up", "fall", "walk", "run", "static"]
STATIC = 6
PAIRS = [("sit<->stand", 0, 1), ("fall<->sit", 3, 0), ("walk<->run", 4, 5), ("pick<->stand", 2, 1)]


def find_runs(root, dataset, variants):
    pat = re.compile(rf"^{re.escape(dataset)}_user_c(\d+)_(.+)$")
    runs = {v: {} for v in variants}
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            m = pat.match(d)
            if m and m.group(2) in runs:
                path = os.path.join(dirpath, d)
                if os.path.exists(os.path.join(path, "test", "predictions.csv")):
                    fold = int(m.group(1))
                    if fold in runs[m.group(2)]:
                        print(f"warning: two runs for {m.group(2)} fold {fold}, using {path}")
                    runs[m.group(2)][fold] = path
    return runs


def read_preds(path):
    with open(os.path.join(path, "test", "predictions.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    return np.array([int(r["true"]) for r in rows]), np.array([int(r["pred"]) for r in rows])


def mcnemar(a_ok, b_ok):
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n, k = n01 + n10, min(n01, n10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n01, n10, p


def fmt_ms(xs):
    xs = np.asarray(xs, dtype=float) * 100
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{xs.mean():.2f} ± {xs.std(ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--dataset", default="sshar_esp")
    ap.add_argument("--out", default=None, help="output folder (default: <root>/summary)")
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "summary")
    os.makedirs(out, exist_ok=True)

    runs = find_runs(args.root, args.dataset, args.variants)
    preds = {v: {f: read_preds(p) for f, p in runs[v].items()} for v in args.variants}
    lines = []

    # ---- accuracy per fold, mean +- std ----
    folds = sorted({f for v in args.variants for f in preds[v]})
    lines.append("## Accuracy (6 actions / 7 classes)\n")
    lines.append("| Variant | " + " | ".join(f"fold {f}" for f in folds) + " | 6 actions | 7 classes |")
    lines.append("|---|" + "---|" * len(folds) + "---|---|")
    for v in args.variants:
        cells, a6, a7 = [], [], []
        for f in folds:
            if f not in preds[v]:
                cells.append("–")
                continue
            t, p = preds[v][f]
            act = t != STATIC
            a6.append(np.mean(p[act] == t[act]))
            a7.append(np.mean(p == t))
            cells.append(f"{100 * a6[-1]:.2f} / {100 * a7[-1]:.2f}")
        lines.append(f"| {v} | " + " | ".join(cells) + f" | {fmt_ms(a6) if a6 else '–'} | {fmt_ms(a7) if a7 else '–'} |")

    # ---- pooled McNemar ----
    lines.append("\n## Pooled McNemar (folds present for both)\n")
    lines.append("| A vs B | folds | 6 actions: A-only / B-only, p | 7 classes: A-only / B-only, p |")
    lines.append("|---|---|---|---|")
    for a, b in combinations(args.variants, 2):
        common = sorted(set(preds[a]) & set(preds[b]))
        if not common:
            continue
        cells = []
        for scope in ("6", "7"):
            ok_a, ok_b = [], []
            for f in common:
                (ta, pa), (tb, pb) = preds[a][f], preds[b][f]
                if not np.array_equal(ta, tb):
                    raise ValueError(f"fold {f}: test labels of {a} and {b} differ (sample order?)")
                keep = ta != STATIC if scope == "6" else np.ones_like(ta, dtype=bool)
                ok_a.append(pa[keep] == ta[keep])
                ok_b.append(pb[keep] == tb[keep])
            n01, n10, p = mcnemar(np.concatenate(ok_a), np.concatenate(ok_b))
            cells.append(f"{n01} / {n10}, p = {p:.4f}")
        lines.append(f"| {a} vs {b} | {','.join(map(str, common))} | " + " | ".join(cells) + " |")

    # ---- pair errors summed over folds ----
    lines.append("\n## Pair errors (sum over folds)\n")
    hdr = [name for name, _, _ in PAIRS] + ["action->static", "all action errors"]
    lines.append("| Variant | folds | " + " | ".join(hdr) + " |")
    lines.append("|---|---|" + "---|" * len(hdr))
    for v in args.variants:
        if not preds[v]:
            continue
        cm = np.zeros((7, 7), dtype=int)
        for t, p in preds[v].values():
            np.add.at(cm, (t, p), 1)
        vals = [cm[i, j] + cm[j, i] for _, i, j in PAIRS]
        vals += [cm[:STATIC, STATIC].sum(), cm[:STATIC].sum() - np.trace(cm[:STATIC, :STATIC])]
        lines.append(f"| {v} | {','.join(map(str, sorted(preds[v])))} | " + " | ".join(map(str, vals)) + " |")

    # ---- gate ----
    for v in args.variants:
        gates = []
        for f, path in sorted(runs[v].items()):
            gp = os.path.join(path, "gate.json")
            if os.path.exists(gp):
                with open(gp, encoding="utf-8") as fh:
                    gates.append(json.load(fh))
        if not gates:
            continue
        cs = sorted({int(c) for g in gates for c in g["by_class_moment"]})
        curves = {c: np.mean([g["by_class_moment"][str(c)] for g in gates if str(c) in g["by_class_moment"]], axis=0)
                  for c in cs}
        lines.append(f"\n## Gate ({v}, {len(gates)} folds): mean share of the shape branch g\n")
        lines.append("| class | mean g |")
        lines.append("|---|---|")
        lines.append(f"| all | {np.mean([g['mean'] for g in gates]):.3f} |")
        for c in cs:
            lines.append(f"| {CLASSES[c]} | {curves[c].mean():.3f} |")
        K = len(next(iter(curves.values())))
        with open(os.path.join(out, f"gate_by_class_moment_{v}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["class"] + [f"k{k}" for k in range(K)])
            for c in cs:
                w.writerow([CLASSES[c]] + [f"{x:.4f}" for x in curves[c]])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4.5))
            for c in cs:
                plt.plot(range(K), curves[c], label=CLASSES[c])
            plt.axhline(0.5, color="gray", lw=0.8, ls="--")
            plt.xlabel("moment k")
            plt.ylabel("g (share of the shape branch)")
            plt.ylim(0, 1)
            plt.title(f"{v}: gate per class and moment (mean over {len(gates)} folds)")
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"gate_by_class_moment_{v}.png"), dpi=200)
            plt.close()
        except ImportError:
            print("matplotlib not found: gate plot skipped")

    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
