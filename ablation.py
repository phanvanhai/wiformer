"""
ablation.py - WiFormer ablation runner

Run every variant (or a subset) through main.py, each one into its own
folder <result_root>/<dataset>_wiformer_<variant>, then collect all
metrics.json into <result_root>/ablation_<dataset>.csv.

Examples
--------
# list variants
python ablation.py --list

# all variants on xrf55 (split by repetition)
python ablation.py --data_root=".\\data\\" --dataset=xrf55

# only a few variants, cross-user case 2, 50 epochs
python ablation.py --data_root=".\\data\\" --dataset=sshar_esp ^
    --variants full k1_time k1_freq --extra --split_mode=user --case_split=2 --epochs=50

# only rebuild the summary table from finished runs
python ablation.py --dataset=xrf55 --summary_only
"""

import argparse
import csv
import json
import os
import subprocess
import sys

# name -> (group, description, extra CLI args for main.py)
# Every variant changes exactly ONE thing w.r.t. "full".
VARIANTS = {
    "full": ("Full", "CSI-WiFormer (default)", []),

    # --- Block 1: Conv1d || FFT ---
    "k1_time": ("Block 1", "Conv1d only (no FFT branch)", ["--k1_branch=time"]),
    "k1_freq": ("Block 1", "FFT only (no Conv1d branch)", ["--k1_branch=freq"]),

    # --- Block 2: Rx x Sub attention ---
    "k2_per_rx": ("Block 2", "Attention per Rx (no cross-Rx), mean over Rx", ["--k2_mode=per_rx"]),
    "k2_pool": ("Block 2", "No Block 2: mean pooling over Rx x Sub", ["--k2_mode=pool"]),
    "k2_pe_none": ("Block 2", "No PE_sub", ["--k2_pe=none"]),
    "k2_pe_rx_sub": ("Block 2", "PE_sub + PE_rx", ["--k2_pe=rx_sub"]),

    # --- Block 3: temporal Transformer ---
    "k3_learned": ("Block 3", "Learned absolute PE instead of ALiBi", ["--k3_pos=learned"]),
    "k3_nopos": ("Block 3", "No positional info (no ALiBi)", ["--k3_pos=none"]),
    "k3_pool": ("Block 3", "No Block 3: mean pooling over windows", ["--k3_layers=0"]),

    # --- Depth ---
    "k2_l1": ("Depth", "Block 2: 1 layer", ["--k2_layers=1"]),
    "k3_l2": ("Depth", "Block 3: 2 layers", ["--k3_layers=2"]),

    # --- Input / preprocessing ---
    "no_norm": ("Input", "No per-sample z-score", ["--no_sample_norm"]),
    "all_ant": ("Input", "All antennas as views (R x A)", ["--ant_mode=all"]),
    "win32": ("Input", "Window 32 / stride 16", ["--window=32", "--stride=16"]),
    "win128": ("Input", "Window 128 / stride 64", ["--window=128", "--stride=64"]),
}

METRIC_KEYS = [
    "accuracy", "f1_macro", "precision_macro", "recall_macro",
    "best_epoch", "total_params", "flops_per_sample", "training_time",
]


def parse_args():
    p = argparse.ArgumentParser(description="WiFormer ablation runner")
    p.add_argument("--dataset", type=str, default=None,
                   choices=["ut_har", "xrf55", "sshar_esp", "sshar_asus"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--result_root", type=str, default="results/ablation")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Subset of variants (default: all)")
    p.add_argument("--skip_done", action="store_true",
                   help="Skip variants whose metrics.json already has accuracy")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --extra is passed to main.py (e.g. --epochs=50)")
    return p.parse_args()


def run_dir(result_root, dataset, name):
    return os.path.join(result_root, f"{dataset}_wiformer_{name}")


def is_done(path):
    f = os.path.join(path, "metrics.json")
    if not os.path.isfile(f):
        return False
    with open(f, encoding="utf-8") as fh:
        return "accuracy" in json.load(fh)


def run_variant(args, name):
    cmd = [
        sys.executable, "main.py",
        f"--data_root={args.data_root}",
        f"--dataset={args.dataset}",
        "--model=wiformer",
        f"--result_root={args.result_root}",
        f"--run_name={name}",
        *VARIANTS[name][2],
        *args.extra,
    ]
    print("\n" + "#" * 70)
    print(f"# [{name}] {VARIANTS[name][1]}")
    print("# " + " ".join(cmd))
    print("#" * 70, flush=True)
    return subprocess.run(cmd).returncode


def summarize(result_root, dataset, names):
    rows = []
    full_acc = None
    for name in names:
        f = os.path.join(run_dir(result_root, dataset, name), "metrics.json")
        if not os.path.isfile(f):
            continue
        with open(f, encoding="utf-8") as fh:
            m = json.load(fh)
        if "accuracy" not in m:
            continue
        group, desc, _ = VARIANTS[name]
        row = {"variant": name, "group": group, "description": desc}
        row.update({k: m.get(k) for k in METRIC_KEYS})
        rows.append(row)
        if name == "full":
            full_acc = m["accuracy"]

    if not rows:
        print("No finished runs found.")
        return

    for r in rows:
        r["delta_acc"] = None if full_acc is None else r["accuracy"] - full_acc

    out = os.path.join(result_root, f"ablation_{dataset}.csv")
    os.makedirs(result_root, exist_ok=True)
    fields = ["variant", "group", "description", *METRIC_KEYS, "delta_acc"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nAblation summary - {dataset}")
    print(f"{'variant':<14}{'group':<9}{'acc':>8}{'Δacc':>8}{'F1':>8}{'params':>10}  description")
    for r in rows:
        d = "" if r["delta_acc"] is None else f"{100 * r['delta_acc']:+.2f}"
        print(
            f"{r['variant']:<14}{r['group']:<9}{100 * r['accuracy']:>8.2f}{d:>8}"
            f"{100 * r['f1_macro']:>8.2f}{r['total_params'] or 0:>10}  {r['description']}"
        )
    print(f"\nSaved: {out}")


def main():
    args = parse_args()

    if args.list:
        for name, (group, desc, cli) in VARIANTS.items():
            print(f"{name:<14}{group:<9}{desc:<48}{' '.join(cli)}")
        return

    if args.dataset is None:
        sys.exit("--dataset is required")

    names = args.variants or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        sys.exit(f"Unknown variants: {unknown}. Use --list.")

    if not args.summary_only:
        if args.data_root is None:
            sys.exit("--data_root is required")
        failed = []
        for name in names:
            if args.skip_done and is_done(run_dir(args.result_root, args.dataset, name)):
                print(f"[skip] {name} already done")
                continue
            if run_variant(args, name) != 0:
                failed.append(name)
        if failed:
            print(f"\nFailed variants: {failed}")

    summarize(args.result_root, args.dataset, names)


if __name__ == "__main__":
    main()
