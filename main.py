"""
main.py

WiFi HAR Benchmark
"""

import argparse
import json
import os
import random
import time

import numpy as np

import torch

from datasets import load_dataset
from models import get_model
from train import train
from metrics import (
    evaluate_model,
    measure_model_complexity,
    save_model_complexity,
)

# ============================================================
# Global seed
# ============================================================
SEED = 42

def set_seed(seed=SEED):
    """
    Set global random seed for reproducibility.

    Covers
    ------
    - Python `random`
    - NumPy
    - PyTorch (CPU + CUDA)
    - cuDNN deterministic mode
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Argument Parser
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="WiFi HAR Benchmark"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "ut_har",
            "xrf55",
            "sshar_esp",
            "sshar_asus",
        ],
        help="Dataset name",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "mlp",
            "cnn5",
            "bilstm",
            "vit",
            "resnet18",
            "resnet1d",
            "cnn_gru",
            "wiformer",
        ],
        help="Model name",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root directory",
    )

    parser.add_argument(
        "--result_root",
        type=str,
        default="results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * accum_steps)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping: stop after N epochs without val-acc improvement (0 = off)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Default: per-model (wiformer=32, others=64)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Default: 1e-3 for all models",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adam", "adamw"],
        help="Default: adam for all models",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Default: 0 for all models",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
        help="Linear warmup epochs before the scheduler (wiformer=5)",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=[
            "none",
            "steplr",
            "multisteplr",
            "cosineannealinglr",
            "reducelronplateau",
            "onecyclelr",
        ],
        help="Optional learning rate scheduler",
    )

    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=30,
        help="Step size for StepLR",
    )

    parser.add_argument(
        "--scheduler-milestones",
        type=str,
        default="",
        help="Comma-separated milestones for MultiStepLR",
    )

    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="Gamma for scheduler decay",
    )

    parser.add_argument(
        "--scheduler-t-max",
        type=int,
        default=None,
        help="T_max for CosineAnnealingLR (default: epochs - warmup)",
    )

    parser.add_argument(
        "--scheduler-eta-min",
        type=float,
        default=0.0,
        help="Eta min for CosineAnnealingLR",
    )

    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=10,
        help="Patience for ReduceLROnPlateau",
    )

    parser.add_argument(
        "--scheduler-threshold",
        type=float,
        default=1e-4,
        help="Threshold for ReduceLROnPlateau",
    )

    parser.add_argument(
        "--scheduler-max-lr",
        type=float,
        default=None,
        help="Max LR for OneCycleLR",
    )

    parser.add_argument(
        "--scheduler-steps-per-epoch",
        type=int,
        default=None,
        help="Steps per epoch for OneCycleLR",
    )

    parser.add_argument(
        "--scheduler-total-steps",
        type=int,
        default=None,
        help="Total steps for OneCycleLR",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=[
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--k2_layers",
        type=int,
        default=2,
        help="WiFormer: number of layers in Block 2 (Rx x Sub attention)",
    )

    parser.add_argument(
        "--k3_layers",
        type=int,
        default=4,
        help="WiFormer: number of layers in Block 3 (temporal Transformer)",
    )

    parser.add_argument(
        "--k1_branch",
        type=str,
        default="both",
        choices=["both", "time", "freq", "none"],
        help="WiFormer Block 1 branches: Conv1d+FFT (both), Conv1d only (time), FFT only (freq), no Block 1 = Linear(W, D) on the raw window (none)",
    )

    # ---- WiFormer ablation knobs (defaults = full model) ----
    parser.add_argument(
        "--k2_mode",
        type=str,
        default="joint",
        choices=["joint", "per_rx", "pool"],
        help="WiFormer Block 2: joint Rx x Sub attention / attention per Rx then mean / mean pooling",
    )
    parser.add_argument(
        "--k2_pe",
        type=str,
        default="sub",
        choices=["sub", "rx_sub", "none"],
        help="WiFormer Block 2 positional embedding",
    )
    parser.add_argument(
        "--k3_pos",
        type=str,
        default="alibi",
        choices=["alibi", "learned", "none"],
        help="WiFormer Block 3 positional encoding (k3_layers=0 -> mean pooling over windows)",
    )
    parser.add_argument(
        "--ant_mode",
        type=str,
        default="first",
        choices=["first", "all"],
        help="WiFormer: keep antenna[0] of each Rx, or use every (rx, antenna) as a view",
    )
    parser.add_argument(
        "--no_sample_norm",
        action="store_true",
        help="WiFormer: disable per-sample z-score",
    )
    # ---- WiFormer capacity (defaults = current model) ----
    parser.add_argument("--d_model", type=int, default=64, help="WiFormer: token dim D (all blocks)")
    parser.add_argument("--feat", type=int, default=64, help="WiFormer Block 1: Conv1d output channels")
    parser.add_argument("--k1_conv", type=int, default=2, choices=[2, 3, 4],
                        help="WiFormer Block 1: number of Conv1d layers")
    parser.add_argument("--k1_pool", type=str, default="avg",
                        help="WiFormer Block 1: 'avg' (global avg) or 'flatN' (keep N steps, e.g. flat4)")
    parser.add_argument("--k2_heads", type=int, default=4, help="WiFormer Block 2: attention heads")
    parser.add_argument("--k2_ff", type=int, default=None, help="WiFormer Block 2: FFN width (default 4*D)")
    parser.add_argument("--k3_heads", type=int, default=4, help="WiFormer Block 3: attention heads")
    parser.add_argument("--k3_ff", type=int, default=None, help="WiFormer Block 3: FFN width (default 4*D)")
    parser.add_argument("--dropout", type=float, default=0.1, help="WiFormer: dropout")
    parser.add_argument("--sub_group", type=int, default=1,
                        help="WiFormer: average every N adjacent subcarriers before Block 1 (56 -> 56/N tokens)")
    parser.add_argument("--window", type=int, default=64,
                        help="WiFormer window W (0 = no windowing: one window over the whole sequence)")
    parser.add_argument("--stride", type=int, default=32, help="WiFormer stride")
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Suffix of the result folder: <dataset>_<model>[_<run_name>]",
    )

    parser.add_argument(
            "--case_split",
            type=int,
            default=0,
            choices=[0, 1, 2, 3, 4],
            help="case split: 0..4 for --split_mode=user, 0..3 for --split_mode=position"
        )

    parser.add_argument(
            "--split_mode",
            type=str,
            default="repeat",
            choices=["repeat", "user", "position"],
            help="split: 'repeat' (SSHAR: protocol 7:1:2 by repetition; XRF55: random 7:1:1), 'user' (cross-user, uses --case_split) "
                 "or 'position' (SSHAR cross-position, --case_split 0..3 holds out pos %% 4 == case)"
        )

    return parser.parse_args()


def get_default_train_cfg(model_name):
    """Per-model defaults for lr / batch size / optimizer."""
    model_name = model_name.lower()
    if model_name == "wiformer":
        # Adam, lr=1e-3 (as in common WiFi-HAR baselines). Warmup + cosine
        # are kept for the Transformer so the large initial lr stays stable.
        return {
            "lr": 1e-3,
            "batch_size": 32,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "warmup_epochs": 5,
        }
    return {
        "lr": 1e-3,
        "batch_size": 64,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "warmup_epochs": 0,
    }


def get_default_scheduler_cfg(model_name, epochs=100, warmup_epochs=0):
    model_name = model_name.lower()
    if model_name == "wiformer":
        return {
            "name": "cosineannealinglr",
            "args": {
                "t_max": max(1, epochs - warmup_epochs),
            },
        }
    if model_name == "resnet1d":
        return {
            "name": "multisteplr",
            "args": {
                "milestones": [40, 80, 120, 160],
                "gamma": 0.5,
            },
        }
    if model_name == "resnet18":
        return {
            "name": "steplr",
            "args": {
                "step_size": 30,
                "gamma": 0.1,
            },
        }
    return None


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    train_defaults = get_default_train_cfg(args.model)
    for key, value in train_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    set_seed(SEED)
    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("WiFi HAR Benchmark")
    print("=" * 60)
    print("Dataset :", args.dataset)
    print("Model   :", args.model)
    print("Device  :", device)
    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    dataset_kwargs = {}
    dataset_kwargs["case_split"] = args.case_split
    dataset_kwargs["split_mode"] = args.split_mode

    if args.model == "cnn_gru":
        train_loader, val_loader, test_loader, info = load_dataset(
            name=args.dataset,
            root_dir=args.data_root,
            batch_size=args.batch_size,
            shape_option="3d",
            intel_shape=True,
            **dataset_kwargs,
        )
    else:
        train_loader, val_loader, test_loader, info = load_dataset(
            name=args.dataset,
            root_dir=args.data_root,
            batch_size=args.batch_size,
            intel_shape=True,
            **dataset_kwargs,
        )

    input_shape = info["input_shape"]
    num_classes = info["num_classes"]
    train_size = info["train_size"]
    val_size = info["val_size"]
    test_size = info["test_size"]

    print("Input Shape :", input_shape)
    print("Classes     :", num_classes)
    print("Train Size  :", train_size)
    print("Val Size    :", val_size)
    print("Test Size   :", test_size)
    print()

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model_kwargs = {}
    if args.model == "wiformer":
        model_kwargs.update(
            k2_layers=args.k2_layers,
            k3_layers=args.k3_layers,
            k1_branch=args.k1_branch,
            k2_mode=args.k2_mode,
            k2_pe=args.k2_pe,
            k3_pos=args.k3_pos,
            ant_mode=args.ant_mode,
            sample_norm=not args.no_sample_norm,
            window=args.window,
            stride=args.stride,
            d_model=args.d_model,
            feat=args.feat,
            k1_conv=args.k1_conv,
            k1_pool=args.k1_pool,
            k2_heads=args.k2_heads,
            k2_ff=args.k2_ff,
            k3_heads=args.k3_heads,
            k3_ff=args.k3_ff,
            dropout=args.dropout,
            sub_group=args.sub_group,
        )

    model = get_model(
        model_name=args.model,
        input_shape=input_shape,
        num_classes=num_classes,
        dataset=args.dataset,
        **model_kwargs,
    )
    model = model.to(device)

    scheduler_cfg = get_default_scheduler_cfg(
        args.model, args.epochs, args.warmup_epochs
    )
    if args.scheduler and args.scheduler.lower() != "none":
        scheduler_cfg = {
            "name": args.scheduler,
            "args": {
                "step_size": args.scheduler_step_size,
                "gamma": args.scheduler_gamma,
                "milestones": args.scheduler_milestones,
                "t_max": (
                    args.scheduler_t_max
                    if args.scheduler_t_max is not None
                    else max(1, args.epochs - args.warmup_epochs)
                ),
                "eta_min": args.scheduler_eta_min,
                "patience": args.scheduler_patience,
                "threshold": args.scheduler_threshold,
                "max_lr": args.scheduler_max_lr,
                "steps_per_epoch": args.scheduler_steps_per_epoch,
                "total_steps": args.scheduler_total_steps,
            },
        }

    if scheduler_cfg is not None:
        scheduler_cfg["warmup_epochs"] = args.warmup_epochs
    optimizer_cfg = {
        "name": args.optimizer,
        "weight_decay": args.weight_decay,
    }
    print(
        f"Optimizer : {args.optimizer} (lr={args.lr}, wd={args.weight_decay})  "
        f"Batch: {args.batch_size} x accum {args.accum_steps}  "
        f"Scheduler: {scheduler_cfg['name'] if scheduler_cfg else 'none'} "
        f"(warmup={args.warmup_epochs})"
    )

    # --------------------------------------------------------
    # Result folder
    # --------------------------------------------------------
    run_dir = f"{args.dataset}_{args.model}"
    if args.run_name:
        run_dir += f"_{args.run_name}"
    save_dir = os.path.join(
        args.result_root,
        run_dir,
    )
    os.makedirs(
        save_dir,
        exist_ok=True,
    )
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)
    # --------------------------------------------------------
    # Count and save model complexity before training
    # --------------------------------------------------------
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    complexity = measure_model_complexity(
        model=model,
        input_shape=input_shape,
    )

    def format_count(value):
        if value is None:
            return "N/A"
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.3f}G"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.3f}M"
        if value >= 1_000:
            return f"{value / 1_000:.3f}K"
        return str(value)

    print("Model Complexity")
    print(f"Total Params     : {format_count(total_params)}")
    print(f"Trainable Params : {format_count(trainable_params)}")
    print(f"MACs / sample    : {format_count(complexity['macs_per_sample'])}")
    print(f"FLOPs / sample   : {format_count(complexity['flops_per_sample'])}")

    save_model_complexity(
        save_dir=save_dir,
        dataset=args.dataset,
        model=args.model,
        total_params=total_params,
        trainable_params=trainable_params,
        complexity=complexity,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    start_time = time.time()
    history, y_true, y_pred, best_epoch, best_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=save_dir,
        scheduler_cfg=scheduler_cfg,
        optimizer_cfg=optimizer_cfg,
        patience=args.patience,
        accum_steps=args.accum_steps,
    )
    training_time = time.time() - start_time
    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    evaluate_model(
        history=history,
        y_true=y_true,
        y_pred=y_pred,
        save_dir=save_dir,
        dataset=args.dataset,
        model=args.model,
        best_epoch=best_epoch,
        train_size=train_size,
        test_size=test_size,
        training_time=training_time,
        total_params=total_params,
        trainable_params=trainable_params,
        complexity=complexity,
        val_size=val_size,
        best_val_acc=best_acc,
        epochs_trained=len(history),
    )

    print()
    print("=" * 60)
    print("Finished")
    print("Result Folder")
    print(save_dir)
    print("=" * 60)

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    main()

# python main.py --data_root=".\data\" --dataset="ut_har" --model="mlp" --case_split=0
# python main.py --data_root=".\data\" --dataset="xrf55" --model="wiformer" --split_mode=user --case_split=2