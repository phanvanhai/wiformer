"""
ablation.py - WiFormer ablation runner

Run every variant (or a subset) through main.py, each one into its own
folder <result_root>/<dataset>_wiformer_<variant>, then collect all
metrics.json into <result_root>/ablation_<dataset>.csv.

Examples
--------
# list variants
python ablation.py --list

# main ablation (remove one block at a time) on xrf55 - default --set core
python ablation.py --data_root=".\\data\\" --dataset=xrf55

# design-choice variants / everything
python ablation.py --data_root=".\\data\\" --dataset=xrf55 --set design
python ablation.py --data_root=".\\data\\" --dataset=xrf55 --set all

# only a few variants, cross-user case 2, 50 epochs
python ablation.py --data_root=".\\data\\" --dataset=sshar_esp ^
    --variants full wo_fft wo_conv --extra --split_mode=user --case_split=2 --epochs=50

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

# ---- Main ablation: remove one block at a time ----
# A removed block is replaced by the simplest op that keeps the tensor shape:
#   Block 1 -> Linear(W, D) on the raw window
#   Block 2 -> mean over the Rx x Sub tokens
#   Block 3 -> mean over the windows
CORE_VARIANTS = {
    "full": ("Full", "CSI-WiFormer (all blocks)", []),
    "wo_fft": ("Block 1", "w/o FFT branch (Conv1d only)", ["--k1_branch=time"]),
    "wo_conv": ("Block 1", "w/o Conv1d branch (FFT only)", ["--k1_branch=freq"]),
    "wo_block1": ("Block 1", "w/o Block 1 (Linear(W,D) on raw window)", ["--k1_branch=none"]),
    "wo_block2": ("Block 2", "w/o Block 2 (mean over Rx x Sub)", ["--k2_mode=pool"]),
    "wo_block3": ("Block 3", "w/o Block 3 (mean over windows)", ["--k3_layers=0"]),
}

# ---- Optional: design choices inside each block ----
DESIGN_VARIANTS = {
    # --- Block 2: Rx x Sub attention ---
    "k2_per_rx": ("Block 2", "Attention per Rx (no cross-Rx), mean over Rx", ["--k2_mode=per_rx"]),
    "k2_pe_none": ("Block 2", "No PE_sub", ["--k2_pe=none"]),
    "k2_pe_rx_sub": ("Block 2", "PE_sub + PE_rx", ["--k2_pe=rx_sub"]),

    # --- Block 3: temporal Transformer ---
    "k3_learned": ("Block 3", "Learned absolute PE instead of ALiBi", ["--k3_pos=learned"]),
    "k3_nopos": ("Block 3", "No positional info (no ALiBi)", ["--k3_pos=none"]),

    # --- Depth ---
    "k2_l1": ("Depth", "Block 2: 1 layer", ["--k2_layers=1"]),
    "k3_l2": ("Depth", "Block 3: 2 layers", ["--k3_layers=2"]),

    # --- Input / preprocessing ---
    "no_norm": ("Input", "No per-sample z-score", ["--no_sample_norm"]),
    "all_ant": ("Input", "All antennas as views (R x A)", ["--ant_mode=all"]),
    "win32": ("Input", "Window 32 / stride 16", ["--window=32", "--stride=16"]),
    "win128": ("Input", "Window 128 / stride 64", ["--window=128", "--stride=64"]),
}

# ---- Optional: scale up (one change at a time, then combined) ----
# FLOPs vs full measured on SSHAR-ESP (3 x 56 x 1000)
D128 = ["--d_model=128", "--k2_heads=8", "--k3_heads=8"]
K1_DEEP = ["--k1_conv=3", "--k1_pool=flat4"]
SCALE_VARIANTS = {
    # Block 1 (cost = FLOPs vs full; D stays 64)
    "k1_flat4": ("Block 1", "Conv1d keeps 4 steps (flat4) instead of global avg", ["--k1_pool=flat4"]),   # 1.0x
    "k1_conv3": ("Block 1", "3 Conv1d layers", ["--k1_conv=3"]),                                   # 1.7x
    "k1_deep": ("Block 1", "3 Conv1d layers + flat4", K1_DEEP),                                     # 1.7x
    "k1_conv4": ("Block 1", "4 Conv1d layers + flat4", ["--k1_conv=4", "--k1_pool=flat4"]),       # 2.4x
    # Block 2
    "k2_l3": ("Block 2", "Block 2: 3 layers", ["--k2_layers=3"]),                                   # 1.2x
    "k2_l4": ("Block 2", "Block 2: 4 layers", ["--k2_layers=4"]),                                   # 1.5x
    # Block 3 (31 tokens -> extra layers are almost free)
    "k3_l6": ("Block 3", "Block 3: 6 layers", ["--k3_layers=6"]),                                   # 1.0x
    "k3_l8": ("Block 3", "Block 3: 8 layers", ["--k3_layers=8"]),                                   # 1.0x
    # Combined, D = 64
    "opt_b": ("Combined", "Option B: 3 Conv1d + flat4, Block 3 6 layers", [*K1_DEEP, "--k3_layers=6"]),         # 1.7x
    "opt_a": ("Combined", "Option A: flat4, Block 3 6 layers", ["--k1_pool=flat4", "--k3_layers=6"]),      # 1.0x
    # Wider (D = 128) -- much heavier, see commit history
    "d128": ("Width", "D = 128, 8 heads", D128),                                                    # 2.4x
    "proposed": ("Combined", "D=128/8 heads, Block 1 feat 128 + 3 conv + flat4, Block 3 6 layers",
                 [*D128, "--feat=128", *K1_DEEP, "--k3_layers=6"]),                                # 6.5x
}

# ---- Windowing ablation on option B (one of the paper's contributions) ----
# Reference = opt_b (W=64, S=32 -> 30 windows with 50% overlap).
OPT_B = [*K1_DEEP, "--k3_layers=6"]
WINDOW_VARIANTS = {
    # whole 5 s recording as ONE window: Block 1 on 1000 samples, Block 2 once,
    # Block 3 sees a single token (so no temporal modelling across windows)
    "b_nowin": ("Window", "Option B without windowing (W = T, 1 window)",
                [*OPT_B, "--window=0"]),                                     # 0.35x vs opt_b
    "b_nooverlap": ("Window", "Option B, windows without overlap (W = S = 64, 15 windows)",
                    [*OPT_B, "--window=64", "--stride=64"]),                                     # 0.5x vs opt_b
}

VARIANTS = {**CORE_VARIANTS, **DESIGN_VARIANTS, **SCALE_VARIANTS, **WINDOW_VARIANTS}
SETS = {
    "core": list(CORE_VARIANTS),
    # "full" is kept so the summary can compute delta_acc
    "design": ["full", *DESIGN_VARIANTS],
    "scale": ["full", *SCALE_VARIANTS],
    # reference first: delta_acc is computed against opt_b here
    "window": ["opt_b", *WINDOW_VARIANTS],
    "all": list(VARIANTS),
}

METRIC_KEYS = [
    "accuracy", "f1_macro", "precision_macro", "recall_macro",
    "best_epoch", "epochs_trained", "total_params", "flops_per_sample", "training_time",
]


def parse_args():
    p = argparse.ArgumentParser(description="WiFormer ablation runner")
    p.add_argument("--dataset", type=str, default=None,
                   choices=["ut_har", "xrf55", "sshar_esp", "sshar_asus"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--result_root", type=str, default="results/ablation")
    p.add_argument("--set", type=str, default="core", choices=list(SETS),
                   help="core = remove one block at a time (default), "
                        "design = design choices inside blocks, "
                        "scale = bigger blocks, window = windowing on option B, "
                        "all = everything")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Explicit list of variants (overrides --set)")
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


def summarize(result_root, dataset, names, tag):
    rows = []
    # delta_acc is measured against "full" if it is in the list, else the first name
    ref = "full" if "full" in names else names[0]
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
        if name == ref:
            full_acc = m["accuracy"]

    if not rows:
        print("No finished runs found.")
        return

    for r in rows:
        r["delta_acc"] = None if full_acc is None else r["accuracy"] - full_acc

    out = os.path.join(result_root, f"ablation_{dataset}_{tag}.csv")
    os.makedirs(result_root, exist_ok=True)
    fields = ["variant", "group", "description", *METRIC_KEYS, "delta_acc"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nAblation summary - {dataset} ({tag})")
    print(f"{'variant':<14}{'group':<9}{'acc':>8}{'d_acc':>8}{'F1':>8}{'params':>10}  description")
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
        for title, table in (("core (remove one block at a time)", CORE_VARIANTS),
                             ("design (choices inside blocks)", DESIGN_VARIANTS),
                             ("scale (bigger blocks)", SCALE_VARIANTS),
                             ("window (windowing on option B, reference opt_b)", WINDOW_VARIANTS)):
            print(f"[{title}]")
            for name, (group, desc, cli) in table.items():
                print(f"  {name:<14}{group:<9}{desc:<48}{' '.join(cli)}")
        return

    if args.dataset is None:
        sys.exit("--dataset is required")

    names = args.variants or SETS[args.set]
    tag = "custom" if args.variants else args.set
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

    summarize(args.result_root, args.dataset, names, tag)


if __name__ == "__main__":
    main()
