"""
main.py

WiFi HAR Benchmark
"""

import argparse
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
        "--batch_size",
        type=int,
        default=None,
        help="Default: per-model (wiformer=32, others=64)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Default: per-model (wiformer=1e-4, others=1e-3)",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adam", "adamw"],
        help="Default: per-model (wiformer=adamw, others=adam)",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Default: per-model (wiformer=0.01, others=0)",
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
            "--case_split",
            type=int,
            default=0,
            choices=[0, 1, 2, 3, 4],
            help="case split from 0 to 4 (used when --split_mode=user)"
        )

    parser.add_argument(
            "--split_mode",
            type=str,
            default="repeat",
            choices=["repeat", "user"],
            help="train/test split: 'repeat' (by repetition index) or 'user' (cross-user, uses --case_split)"
        )

    return parser.parse_args()


def get_default_train_cfg(model_name):
    """Per-model defaults for lr / batch size / optimizer."""
    model_name = model_name.lower()
    if model_name == "wiformer":
        return {
            "lr": 1e-4,
            "batch_size": 32,
            "optimizer": "adamw",
            "weight_decay": 0.01,
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
        train_loader, test_loader, info = load_dataset(
            name=args.dataset,
            root_dir=args.data_root,
            batch_size=args.batch_size,
            shape_option="3d",
            intel_shape=True,
            **dataset_kwargs,
        )
    else:
        train_loader, test_loader, info = load_dataset(
            name=args.dataset,
            root_dir=args.data_root,
            batch_size=args.batch_size,
            intel_shape=True,
            **dataset_kwargs,
        )

    input_shape = info["input_shape"]
    num_classes = info["num_classes"]
    train_size = info["train_size"]
    test_size = info["test_size"]

    print("Input Shape :", input_shape)
    print("Classes     :", num_classes)
    print("Train Size  :", train_size)
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
        f"Batch: {args.batch_size}  "
        f"Scheduler: {scheduler_cfg['name'] if scheduler_cfg else 'none'} "
        f"(warmup={args.warmup_epochs})"
    )

    # --------------------------------------------------------
    # Result folder
    # --------------------------------------------------------
    save_dir = os.path.join(
        args.result_root,
        f"{args.dataset}_{args.model}",
    )
    os.makedirs(
        save_dir,
        exist_ok=True,
    )
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
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=save_dir,
        scheduler_cfg=scheduler_cfg,
        optimizer_cfg=optimizer_cfg,
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