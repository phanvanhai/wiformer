"""
train.py

Simple training utilities
"""

import inspect
import os
import time
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import lr_scheduler


SCHEDULER_ARG_KEYS = {
    "steplr": {"step_size": "step_size", "gamma": "gamma"},
    "multisteplr": {"milestones": "milestones", "gamma": "gamma"},
    "cosineannealinglr": {"t_max": "T_max", "T_max": "T_max", "eta_min": "eta_min"},
    "reducelronplateau": {"patience": "patience", "threshold": "threshold", "gamma": "factor"},
    "onecyclelr": {
        "max_lr": "max_lr",
        "steps_per_epoch": "steps_per_epoch",
        "total_steps": "total_steps",
        "epochs": "epochs",
    },
}


def create_scheduler(optimizer, scheduler_cfg=None):
    """
    Create a learning-rate scheduler from a config dict.

    scheduler_cfg = {
        "name": "cosineannealinglr",
        "args": {...},              # only keys relevant to `name` are used
        "warmup_epochs": 5,         # optional linear warmup (epoch-level)
    }
    """
    if scheduler_cfg is None:
        return None

    scheduler_name = scheduler_cfg.get("name")
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None
    scheduler_name = scheduler_name.lower()
    if scheduler_name not in SCHEDULER_ARG_KEYS:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}.")

    raw_args = scheduler_cfg.get("args", {}) or {}
    key_map = SCHEDULER_ARG_KEYS[scheduler_name]
    scheduler_args = {
        key_map[key]: value
        for key, value in raw_args.items()
        if key in key_map and value is not None
    }

    if scheduler_name == "steplr":
        scheduler = lr_scheduler.StepLR(optimizer, **scheduler_args)
    elif scheduler_name == "multisteplr":
        milestones = scheduler_args.get("milestones")
        if isinstance(milestones, str):
            scheduler_args["milestones"] = [
                int(x)
                for x in milestones.split(",")
                if x.strip()
            ]
        scheduler = lr_scheduler.MultiStepLR(optimizer, **scheduler_args)
    elif scheduler_name == "cosineannealinglr":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
    elif scheduler_name == "reducelronplateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)
    else:
        return lr_scheduler.OneCycleLR(optimizer, **scheduler_args)

    warmup_epochs = scheduler_cfg.get("warmup_epochs") or 0
    if warmup_epochs > 0:
        warmup = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / (warmup_epochs + 1),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, scheduler],
            milestones=[warmup_epochs],
        )
    return scheduler


def create_optimizer(model, lr, optimizer_cfg=None):
    """Create optimizer. optimizer_cfg = {"name": "adam"|"adamw", "weight_decay": float}."""
    optimizer_cfg = optimizer_cfg or {}
    name = (optimizer_cfg.get("name") or "adam").lower()
    weight_decay = optimizer_cfg.get("weight_decay") or 0.0
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}.")


# ============================================================
# Train one epoch
# ============================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """
    Train one epoch.

    Returns
    -------
    train_loss
    train_acc
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / len(loader)
    train_acc = correct / total

    return train_loss, train_acc

# ============================================================
# Evaluate
# ============================================================
@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate model.

    Returns
    -------
    loss
    accuracy
    y_true
    y_pred
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        y_true.extend(
            y.cpu().numpy()
        )

        y_pred.extend(
            pred.cpu().numpy()
        )

    test_loss = running_loss / len(loader)
    test_acc = correct / total

    return (
        test_loss,
        test_acc,
        y_true,
        y_pred,
    )

# ============================================================
# Train
# ============================================================

def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    lr,
    device,
    save_dir,
    scheduler_cfg=None,
    optimizer_cfg=None,
    patience=10,
):
    """
    Train model.

    Parameters
    ----------
    scheduler_cfg : dict or None
        Optional scheduler configuration dict with keys:
        - name: scheduler type
        - args: scheduler arguments
        - warmup_epochs: optional linear warmup
    optimizer_cfg : dict or None
        {"name": "adam" | "adamw", "weight_decay": float}
    patience : int
        Early stopping: stop when val acc has not improved for `patience`
        epochs in a row. 0 or None disables it.

    Returns
    -------
    history
    y_true
    y_pred
    """

    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, lr, optimizer_cfg)
    scheduler = create_scheduler(optimizer, scheduler_cfg)

    history = []

    best_acc = -1.0  # ensure best_model.pth is saved at least once
    best_epoch = 0
    epochs_no_improve = 0

    start_time = time.time()
    # --------------------------------------------------------
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"[{epoch:03d}/{epochs}] "
            f"Train Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  "
            f"Val Acc: {val_acc:.4f}"
        )

        # ----------------------------------------------------
        # Save best model (selected on VAL, never on test)
        # ----------------------------------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_dir,
                    "best_model.pth",
                ),
            )

        if scheduler is not None:
            if isinstance(
                scheduler,
                lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ----------------------------------------------------
        # Early stopping (on val acc)
        # ----------------------------------------------------
        if epoch != best_epoch:
            epochs_no_improve += 1
        if patience and epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch}: val acc has not improved "
                f"for {patience} epochs (best {best_acc:.4f} @ epoch {best_epoch})"
            )
            break

    # --------------------------------------------------------
    # Training finished
    # --------------------------------------------------------
    elapsed = time.time() - start_time

    print("\nTraining Finished")
    print(f"Best Epoch : {best_epoch}")
    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Time       : {elapsed:.1f} sec")

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------
    history = pd.DataFrame(history)
    history.to_csv(
        os.path.join(
            save_dir,
            "history.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------
    model.load_state_dict(
        torch.load(
            os.path.join(
                save_dir,
                "best_model.pth",
            ),
            map_location=device,
        )
    )

    # --------------------------------------------------------
    # Final evaluation on TEST (best-on-val checkpoint)
    # --------------------------------------------------------
    test_loss, test_acc, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    print(f"Test Acc   : {test_acc:.4f}")

    return (
        history,
        y_true,
        y_pred,
        best_epoch,
        best_acc,
    )