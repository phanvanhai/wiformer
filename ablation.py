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

# WiFormer v4 core ablation (every split is 6:2:2)
# result folders: <result_root>/<dataset>_<split>_<variant>, e.g. sshar_esp_repeat_full,
#                 ut_har_official_nonorm, sshar_esp_user_c0_full
python ablation.py --model=wiformer_v4 --data_root=".\\data\\" --dataset=sshar_esp
python ablation.py --model=wiformer_v4 --data_root=".\\data\\" --dataset=ut_har --variants nonorm
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
    "no_norm": ("Input", "No per-sample normalisation (loader global z-score)", ["--no_sample_norm"]),
    "all_ant": ("Input", "All antennas as views (R x A)", ["--ant_mode=all"]),
    "win32": ("Input", "Window 32 / stride 16", ["--window=32", "--stride=16"]),
    "win128": ("Input", "Window 128 / stride 64", ["--window=128", "--stride=64"]),
}

# ---- Optional: scale up (one change at a time, then combined) ----
# These were run BEFORE the current default (3 Conv1d + flat4, Block 3 6 layers), so each is pinned to the
# old base (2 Conv1d + avg pool, Block 3 4 layers) to keep its meaning.
# FLOPs vs the old base measured on SSHAR-ESP (3 x 56 x 1000)
OLD = ["--k1_conv=2", "--k1_pool=avg", "--k3_layers=4"]
D128 = ["--d_model=128", "--k2_heads=8", "--k3_heads=8"]
K1_DEEP = ["--k1_conv=3", "--k1_pool=flat4"]
SCALE_VARIANTS = {
    "old_full": ("Old base", "Old default: 2 Conv1d + avg, Block 3 4 layers", OLD),                  # 1.0x
    # Block 1 (D stays 64)
    "k1_flat4": ("Block 1", "Conv1d keeps 4 steps (flat4) instead of global avg", [*OLD, "--k1_pool=flat4"]),  # 1.0x
    "k1_conv3": ("Block 1", "3 Conv1d layers", [*OLD, "--k1_conv=3"]),                                      # 1.7x
    "k1_deep": ("Block 1", "3 Conv1d layers + flat4", [*OLD, *K1_DEEP]),                                    # 1.7x
    "k1_conv4": ("Block 1", "4 Conv1d layers + flat4", [*OLD, "--k1_conv=4", "--k1_pool=flat4"]),          # 2.4x
    # Block 2
    "k2_l3": ("Block 2", "Block 2: 3 layers", [*OLD, "--k2_layers=3"]),                                      # 1.2x
    "k2_l4": ("Block 2", "Block 2: 4 layers", [*OLD, "--k2_layers=4"]),                                      # 1.5x
    # Block 3 (31 tokens -> extra layers are almost free)
    "k3_l6": ("Block 3", "Block 3: 6 layers", [*OLD, "--k3_layers=6"]),                                      # 1.0x
    "k3_l8": ("Block 3", "Block 3: 8 layers", [*OLD, "--k3_layers=8"]),                                      # 1.0x
    # Combined, D = 64
    "opt_b": ("Combined", "3 Conv1d + flat4, Block 3 6 layers (= full)", [*K1_DEEP, "--k3_layers=6"]),             # 1.7x
    "opt_a": ("Combined", "Option A: flat4, Block 3 6 layers", [*OLD, "--k1_pool=flat4", "--k3_layers=6"]),  # 1.0x
    # Wider (D = 128) -- much heavier, see commit history
    "d128": ("Width", "D = 128, 8 heads", [*OLD, *D128]),                                                    # 2.4x
    "proposed": ("Combined", "D=128/8 heads, Block 1 feat 128 + 3 conv + flat4, Block 3 6 layers",
                 [*D128, "--feat=128", *K1_DEEP, "--k3_layers=6"]),                                        # 6.5x
}

# ---- Windowing ablation (one of the paper's contributions) ----
# Reference = full (W=64, S=32 -> 30 windows with 50% overlap).
WINDOW_VARIANTS = {
    # whole 5 s recording as ONE window: Block 1 on 1000 samples, Block 2 once,
    # Block 3 sees a single token (so no temporal modelling across windows)
    "b_nowin": ("Window", "No windowing (W = T, 1 window)", ["--window=0"]),                   # 0.35x vs full
    "b_nooverlap": ("Window", "Windows without overlap (W = S = 64, 15 windows)",
                    ["--window=64", "--stride=64"]),                                             # 0.5x vs full
}

VARIANTS = {**CORE_VARIANTS, **DESIGN_VARIANTS, **SCALE_VARIANTS, **WINDOW_VARIANTS}
SETS = {
    "core": list(CORE_VARIANTS),
    # "full" is kept so the summary can compute delta_acc
    "design": ["full", *DESIGN_VARIANTS],
    "scale": ["full", *SCALE_VARIANTS],
    "window": ["full", *WINDOW_VARIANTS],
    "all": list(VARIANTS),
}

# ---- WiFormer v4 (--model=wiformer_v4): core ablation ----
# see WiFormer_v4_Architecture.md, section 6
V4_VARIANTS = {
    "full": ("Full", "WiFormer v4 (token = [Conv || FFT], relative-bias Blocks 2-3)", ["--v4_ablation=full"]),
    "wo_fft": ("Block 1", "Block 1 = time view only (Conv)", ["--v4_ablation=wo_fft"]),
    "wo_conv": ("Block 1", "Block 1 = spectral view only (FFT)", ["--v4_ablation=wo_conv"]),
    "wo_block1": ("Block 1", "w/o Block 1 (Linear(W,D) on raw window)", ["--v4_ablation=wo_block1"]),
    "wo_block2": ("Block 2", "w/o Block 2 (mean over Rx x Sub)", ["--v4_ablation=wo_block2"]),
    "wo_block3": ("Block 3", "w/o Block 3 (mean over windows)", ["--v4_ablation=wo_block3"]),
    # spectral view only + mean over Rx x Sub (= averaged spectrum); compare with wo_conv
    "freq_mean": ("Block 2", "Spectral view only, mean over Rx x Sub (averaged spectrum)",
                  ["--v4_ablation=freq_mean"]),
    # two branches per window: Conv tokens -> Block 2, and the averaged spectrum; fused before Block 3
    "hybrid": ("Fusion", "[Conv -> Block 2] + averaged spectrum, fused per window",
               ["--v4_ablation=hybrid"]),
    # hybrid + modality dropout: zero the Block 2 branch with p = 0.5 during training
    "hybrid_drop": ("Fusion", "hybrid + modality dropout on h_k (p = 0.5, training only)",
                    ["--v4_ablation=hybrid", "--mod_drop=0.5"]),
    # freq_mean trained 200 epochs (cosine to 0 at 200): has freq_mean converged at 100?
    "freq_mean_e200": ("Block 2", "freq_mean, 200 epochs (EXTRA must not set --epochs)",
                       ["--v4_ablation=freq_mean", "--epochs=200"]),
    # UT_HAR: no per-sample normalisation inside the model (keep the static level)
    "nonorm": ("Norm", "No per-sample normalisation in the model (keep static level)",
               ["--v4_ablation=full", "--norm=none"]),
}
V4_CORE = ["full", "wo_fft", "wo_conv", "wo_block1", "wo_block2", "wo_block3"]
V4_SETS = {"core": V4_CORE, "all": list(V4_VARIANTS)}

# ---- WiFormer v5 (--model=wiformer_v5) ----
V5_VARIANTS = {
    "full": ("Full", "WiFormer v5 (shape: Conv -> Block 2; speed: averaged Hann spectrum; mod_drop 0.5)",
             ["--v5_ablation=full"]),
    "gate": ("Gate fusion", "Scalar gate per moment instead of the linear fusion (mod_drop kept)",
             ["--v5_ablation=full", "--fusion=gate"]),
    "wo_shape": ("w/o shape", "Speed branch only (averaged Hann spectrum -> Linear -> Block 3)",
                 ["--v5_ablation=wo_shape"]),
}
V5_SETS = {"core": list(V5_VARIANTS), "all": list(V5_VARIANTS)}


def tables(model):
    """(variants, sets) of the given model."""
    if model == "wiformer_v4":
        return V4_VARIANTS, V4_SETS
    if model == "wiformer_v5":
        return V5_VARIANTS, V5_SETS
    return VARIANTS, SETS

METRIC_KEYS = [
    "accuracy", "f1_macro", "precision_macro", "recall_macro",
    "best_epoch", "epochs_trained", "total_params", "flops_per_sample", "training_time",
]


def parse_args():
    p = argparse.ArgumentParser(description="WiFormer ablation runner")
    p.add_argument("--dataset", type=str, default=None,
                   choices=["ut_har", "xrf55", "sshar_esp", "sshar_asus"])
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--model", type=str, default="wiformer", choices=["wiformer", "wiformer_v4", "wiformer_v5"])
    p.add_argument("--result_root", type=str, default=None,
                   help="default: results/ablation (wiformer), results/ablation_v4 (wiformer_v4)")
    p.add_argument("--set", type=str, default="core", choices=list(SETS),
                   help="core = remove one block at a time (default), "
                        "design = design choices inside blocks, "
                        "scale = bigger blocks, window = windowing (no window / no overlap), "
                        "all = everything")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Explicit list of variants (overrides --set)")
    p.add_argument("--skip_done", action="store_true",
                   help="Skip variants whose metrics.json already has accuracy")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--split_mode", type=str, default=None,
                   choices=["repeat", "user", "position"],
                   help="all 6:2:2; default repeat; ignored for ut_har")
    p.add_argument("--case_split", type=int, default=0, help="case for --split_mode user / position")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --extra is passed to main.py (e.g. --epochs=50)")
    return p.parse_args()


def split_label(args):
    """Split part of the folder / summary names."""
    if args.dataset == "ut_har":
        return "official"                       # UT_HAR's own train / val / test files
    if args.split_mode in ("user", "position"):
        return f"{args.split_mode}_c{args.case_split}"
    return args.split_mode


def folder_name(args, name):
    """<dataset>_<split>_<variant>."""
    return f"{args.dataset}_{split_label(args)}_{name}"


def run_dir(args, name):
    return os.path.join(args.result_root, folder_name(args, name))


def is_done(path):
    f = os.path.join(path, "metrics.json")
    if not os.path.isfile(f):
        return False
    with open(f, encoding="utf-8") as fh:
        return "accuracy" in json.load(fh)


def run_variant(args, name):
    variants, _ = tables(args.model)
    cmd = [
        sys.executable, "-u", "main.py",      # -u: unbuffered, log appears line by line
        f"--data_root={args.data_root}",
        f"--dataset={args.dataset}",
        f"--model={args.model}",
        f"--result_root={args.result_root}",
        f"--run_dir={folder_name(args, name)}",
        f"--split_mode={args.split_mode}",
        f"--case_split={args.case_split}",
        *variants[name][2],
        *args.extra,          # last -> overrides the defaults above
    ]
    print("\n" + "#" * 70)
    print(f"# [{name}] {variants[name][1]}")
    print("# " + " ".join(cmd))
    print("#" * 70, flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return subprocess.run(cmd, env=env).returncode


def summarize(args, names, tag):
    variants, _ = tables(args.model)
    result_root, dataset, model = args.result_root, args.dataset, args.model
    rows = []
    # delta_acc is measured against "full" if it is in the list, else the first name
    ref = "full" if "full" in names else names[0]
    full_acc = None
    for name in names:
        d = run_dir(args, name)
        f = os.path.join(d, "metrics.json")
        if not os.path.isfile(f):
            continue
        with open(f, encoding="utf-8") as fh:
            m = json.load(fh)
        if "accuracy" not in m:
            continue
        group, desc, _ = variants[name]
        row = {"variant": name, "group": group, "description": desc, "folder": os.path.basename(d)}
        row.update({k: m.get(k) for k in METRIC_KEYS})
        # train / val accuracy of the best checkpoint (summary_splits.csv, newer runs only)
        s = os.path.join(d, "summary_splits.csv")
        if os.path.isfile(s):
            with open(s, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r["split"] in ("train", "val"):
                        row[f"{r['split']}_acc"] = float(r["accuracy"])
                        row[f"{r['split']}_f1"] = float(r["f1_macro"])
        rows.append(row)
        if name == ref:
            full_acc = m["accuracy"]

    if not rows:
        print("No finished runs found.")
        return

    for r in rows:
        r["delta_acc"] = None if full_acc is None else r["accuracy"] - full_acc

    if model == "wiformer":
        out = os.path.join(result_root, f"ablation_{dataset}_{tag}.csv")
    else:
        out = os.path.join(result_root, f"ablation_{model}_{dataset}_{split_label(args)}_{tag}.csv")
    os.makedirs(result_root, exist_ok=True)
    fields = ["variant", "group", "description", "folder", *METRIC_KEYS,
              "train_acc", "train_f1", "val_acc", "val_f1", "delta_acc"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
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
    if args.result_root is None:
        args.result_root = {"wiformer_v4": "results/ablation_v4",
                            "wiformer_v5": "results/ablation_v5"}.get(args.model, "results/ablation")
    if args.split_mode is None:
        args.split_mode = "repeat"
    variants, sets = tables(args.model)

    if args.list:
        groups = ((("v5", V5_VARIANTS),) if args.model == "wiformer_v5" else
                  (("v4 core (Block 1 views / remove one block)", V4_VARIANTS),) if args.model == "wiformer_v4" else
                  (("core (remove one block at a time)", CORE_VARIANTS),
                   ("design (choices inside blocks)", DESIGN_VARIANTS),
                   ("scale (bigger blocks)", SCALE_VARIANTS),
                   ("window (windowing, reference full)", WINDOW_VARIANTS)))
        for title, table in groups:
            print(f"[{title}]")
            for name, (group, desc, cli) in table.items():
                print(f"  {name:<14}{group:<9}{desc:<48}{' '.join(cli)}")
        return

    if args.dataset is None:
        sys.exit("--dataset is required")

    if args.set not in sets:
        sys.exit(f"--set {args.set} is not available for {args.model}: {list(sets)}")
    names = args.variants or sets[args.set]
    tag = "custom" if args.variants else args.set
    unknown = [n for n in names if n not in variants]
    if unknown:
        sys.exit(f"Unknown variants: {unknown}. Use --list.")

    if not args.summary_only:
        if args.data_root is None:
            sys.exit("--data_root is required")
        failed = []
        for name in names:
            if args.skip_done and is_done(run_dir(args, name)):
                print(f"[skip] {name} already done")
                continue
            if run_variant(args, name) != 0:
                failed.append(name)
        if failed:
            print(f"\nFailed variants: {failed}")

    summarize(args, names, tag)


if __name__ == "__main__":
    main()
