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


def create_scheduler(optimizer, scheduler_cfg=None):
    """Create a learning-rate scheduler from a config dict."""
    if scheduler_cfg is None:
        return None

    scheduler_name = scheduler_cfg.get("name")
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None

    scheduler_args = scheduler_cfg.get("args", {}) or {}
    scheduler_args = {
        key: value
        for key, value in scheduler_args.items()
        if value is not None
    }

    scheduler_name = scheduler_name.lower()
    if scheduler_name == "steplr":
        return lr_scheduler.StepLR(optimizer, **scheduler_args)

    if scheduler_name == "multisteplr":
        milestones = scheduler_args.get("milestones")
        if isinstance(milestones, str):
            scheduler_args["milestones"] = [
                int(x)
                for x in milestones.split(",")
                if x.strip()
            ]
        return lr_scheduler.MultiStepLR(optimizer, **scheduler_args)

    if scheduler_name == "cosineannealinglr":
        return lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)

    if scheduler_name == "reducelronplateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

    if scheduler_name == "onecyclelr":
        return lr_scheduler.OneCycleLR(optimizer, **scheduler_args)

    raise ValueError(
        f"Unsupported scheduler: {scheduler_name}."
    )


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
    test_loader,
    epochs,
    lr,
    device,
    save_dir,
    scheduler_cfg=None,
):
    """
    Train model.

    Parameters
    ----------
    scheduler_cfg : dict or None
        Optional scheduler configuration dict with keys:
        - name: scheduler type
        - args: scheduler arguments

    Returns
    -------
    history
    y_true
    y_pred
    """

    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    scheduler = create_scheduler(optimizer, scheduler_cfg)

    history = []

    best_acc = 0.0
    best_epoch = 0

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

        test_loss, test_acc, y_true, y_pred = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

        print(
            f"[{epoch:03d}/{epochs}] "
            f"Train Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.4f}  "
            f"Test Loss: {test_loss:.4f}  "
            f"Test Acc: {test_acc:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
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
                scheduler.step(test_loss)
            else:
                scheduler.step()

    # --------------------------------------------------------
    # Training finished
    # --------------------------------------------------------
    elapsed = time.time() - start_time

    print("\nTraining Finished")
    print(f"Best Epoch : {best_epoch}")
    print(f"Best Acc   : {best_acc:.4f}")
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
    # Final evaluation
    # --------------------------------------------------------
    test_loss, test_acc, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    return (
        history,
        y_true,
        y_pred,
        best_epoch,
        best_acc,
    )