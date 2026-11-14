"""
metrics.py

Evaluation utilities
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

try:
    from thop import profile
except ImportError:
    profile = None

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def measure_model_complexity(model, input_shape):
    """Measure MACs and FLOPs for one inference sample."""

    result = {
        "macs": None,
        "flops": None,
        "macs_per_sample": None,
        "flops_per_sample": None,
    }

    if profile is None:
        print("Warning: install thop to measure MACs/FLOPs.")
        return result

    was_training = model.training
    device = next(model.parameters()).device
    dummy_input = torch.zeros(
        (1, *input_shape),
        device=device,
    )

    try:
        model.eval()
        with torch.no_grad():
            macs, _ = profile(
                model,
                inputs=(dummy_input,),
                verbose=False,
            )
        flops = 2 * macs
        result.update({
            "macs": int(macs),
            "flops": int(flops),
            "macs_per_sample": int(macs),
            "flops_per_sample": int(flops),
        })
    except Exception as exc:
        print(
            "Warning: could not measure MACs/FLOPs "
            f"for this model: {exc}"
        )
    finally:
        model.train(was_training)

    return result


# ============================================================
# Compute Metrics
# ============================================================
def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.

    Returns
    -------
    metrics : dict
    cm : ndarray
    report : dict
    """

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "precision_weighted": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "recall_weighted": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return metrics, cm, report


# ============================================================
# Save metrics.json
# ============================================================
def save_metrics(
    save_dir,
    metrics,
    dataset,
    model,
    best_epoch,
    train_size,
    test_size,
    training_time,
    total_params=None,
    trainable_params=None,
    complexity=None,
    val_size=None,
    best_val_acc=None,
    epochs_trained=None,
):

    data = {
        "dataset": dataset,
        "model": model,
        "accuracy":
            float(metrics["accuracy"]),
        "precision_macro":
            float(metrics["precision_macro"]),
        "recall_macro":
            float(metrics["recall_macro"]),
        "f1_macro":
            float(metrics["f1_macro"]),
        "precision_weighted":
            float(metrics["precision_weighted"]),
        "recall_weighted":
            float(metrics["recall_weighted"]),
        "f1_weighted":
            float(metrics["f1_weighted"]),
        "best_epoch":
            int(best_epoch),
        "epochs_trained":
            None if epochs_trained is None else int(epochs_trained),
        "train_size":
            int(train_size),
        "val_size":
            None if val_size is None else int(val_size),
        "test_size":
            int(test_size),
        "best_val_acc":
            None if best_val_acc is None else float(best_val_acc),
        "training_time":
            float(training_time),
        "total_params":
            total_params,
        "trainable_params":
            trainable_params,
        "macs":
            None if complexity is None else complexity.get("macs"),
        "flops":
            None if complexity is None else complexity.get("flops"),
        "macs_per_sample":
            None if complexity is None else complexity.get("macs_per_sample"),
        "flops_per_sample":
            None if complexity is None else complexity.get("flops_per_sample"),
    }

    with open(
        os.path.join(
            save_dir,
            "metrics.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
        )


def save_model_complexity(
    save_dir,
    dataset,
    model,
    total_params,
    trainable_params,
    complexity,
):
    """Save model complexity before the potentially long training run."""

    data = {
        "dataset": dataset,
        "model": model,
        "status": "pending_training",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "macs": complexity.get("macs"),
        "flops": complexity.get("flops"),
        "macs_per_sample": complexity.get("macs_per_sample"),
        "flops_per_sample": complexity.get("flops_per_sample"),
    }

    with open(
        os.path.join(save_dir, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(data, f, indent=4)


# ============================================================
# Save predictions
# ============================================================
def save_predictions(
    save_dir,
    y_true,
    y_pred,
):

    df = pd.DataFrame({
        "sample":
            np.arange(
                len(y_true)
            ),
        "true":
            y_true,
        "pred":
            y_pred,
    })

    df.to_csv(
        os.path.join(
            save_dir,
            "predictions.csv",
        ),
        index=False,

    )


# ============================================================
# Save classification report
# ============================================================
def save_classification_report(
    save_dir,
    report,
):

    df = pd.DataFrame(
        report
    ).transpose()

    df.to_csv(
        os.path.join(
            save_dir,
            "classification_report.csv",
        ),

        index=True,

    )


# ============================================================
# Save confusion matrix csv
# ============================================================
def save_confusion_matrix(
    save_dir,
    cm,
    class_names=None,
):

    if class_names is None:
        class_names = [
            str(i)
            for i in range(
                len(cm)
            )
        ]

    df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    df.to_csv(
        os.path.join(
            save_dir,
            "confusion_matrix.csv",
        )

    )

    # ============================================================
# Learning Curve
# ============================================================
def plot_learning_curve(
    history,
    save_dir,
):
    """
    Plot learning curves.

    Parameters
    ----------
    history
        pandas.DataFrame
        hoặc history.csv
    """

    if isinstance(history, str):
        history = pd.read_csv(history)

    # new runs log "val_*"; old history.csv files only have "test_*"
    ev, ev_label = ("val", "Val") if "val_loss" in history else ("test", "Test")

    plt.figure(figsize=(10, 4))
    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------
    plt.subplot(1, 2, 1)

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train",
        linewidth=2,
    )

    plt.plot(
        history["epoch"],
        history[f"{ev}_loss"],
        label=ev_label,
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True)
    plt.legend()

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------
    plt.subplot(1, 2, 2)
    plt.plot(
        history["epoch"],
        history["train_acc"],
        label="Train",
        linewidth=2,
    )

    plt.plot(
        history["epoch"],
        history[f"{ev}_acc"],
        label=ev_label,
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir,
            "learning_curve.png",
        ),
        dpi=300,
    )
    plt.close()


# ============================================================
# Plot Confusion Matrix
# ============================================================
def plot_confusion_matrix(
    save_dir,
    cm,
    class_names=None,
):

    if class_names is None:
        class_names = [
            str(i)
            for i in range(
                len(cm)
            )
        ]

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir,
            "confusion_matrix.png",
        ),
        dpi=300,
    )

    plt.close()


# ============================================================
# Per-split report (train / val / test) with the best checkpoint
# ============================================================
# Class names used in reports / figures (index = label after remapping)
CLASS_NAMES = {
    "sshar_esp": ["sit_down", "stand_up", "pick_up", "fall", "lie_still", "walk", "run", "empty"],
}


def _plot_cm(cm, cm_pct, class_names, title, path):
    """Heatmap coloured by row %, each cell annotated 'count\\n(xx.x%)'."""
    annot = np.array([[f"{c}\n({p:.1f}%)" for c, p in zip(rc, rp)] for rc, rp in zip(cm, cm_pct)])
    size = max(6, 0.9 * len(cm) + 2)
    plt.figure(figsize=(size + 1, size))
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=100,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "% of true class (row)"}, annot_kws={"fontsize": 8})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


@torch.no_grad()
def save_split_reports(model, loaders, device, save_dir, dataset=None, num_classes=None):
    """
    Evaluate `model` (already loaded with the best checkpoint) on every split
    and save, for each split in `loaders` (e.g. {"train": ..., "val": ..., "test": ...}):

        <save_dir>/<split>/metrics.json                 accuracy, precision / recall / F1 (macro + weighted), n
        <save_dir>/<split>/classification_report.csv    per-class precision / recall / F1 / support
        <save_dir>/<split>/confusion_matrix_count.csv   number of samples (row = true, col = predicted)
        <save_dir>/<split>/confusion_matrix_rowpct.csv  % of each true class (rows sum to 100)
        <save_dir>/<split>/confusion_matrix.png         counts + row %
        <save_dir>/<split>/predictions.csv              true / pred / per-class probabilities
    and <save_dir>/summary_splits.csv with one row per split.
    """
    names = CLASS_NAMES.get((dataset or "").lower())
    was_training = model.training
    model.eval()
    rows = []
    for split, loader in loaders.items():
        y_true, y_pred, probs = [], [], []
        for x, y in loader:
            logits = model(x.to(device))
            probs.append(torch.softmax(logits.float(), dim=1).cpu())
            y_pred.append(logits.argmax(dim=1).cpu())
            y_true.append(y.cpu())
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        probs = torch.cat(probs).numpy()
        n_cls = num_classes or probs.shape[1]
        labels = list(range(n_cls))
        cls_names = names if names and len(names) == n_cls else [str(i) for i in labels]

        metrics, _, _ = compute_metrics(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(100.0 * cm, row_sum, out=np.zeros(cm.shape), where=row_sum > 0)
        report = classification_report(y_true, y_pred, labels=labels, target_names=cls_names,
                                       output_dict=True, zero_division=0)

        out = os.path.join(save_dir, split)
        os.makedirs(out, exist_ok=True)
        m = {k: float(v) for k, v in metrics.items()}
        m["n"] = int(len(y_true))
        with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, indent=4)
        pd.DataFrame(report).transpose().to_csv(os.path.join(out, "classification_report.csv"))
        pd.DataFrame(cm, index=cls_names, columns=cls_names).to_csv(
            os.path.join(out, "confusion_matrix_count.csv"))
        pd.DataFrame(cm_pct.round(2), index=cls_names, columns=cls_names).to_csv(
            os.path.join(out, "confusion_matrix_rowpct.csv"))
        _plot_cm(cm, cm_pct, cls_names,
                 f"{split} - acc {100 * m['accuracy']:.2f}%  F1 {100 * m['f1_macro']:.2f}%",
                 os.path.join(out, "confusion_matrix.png"))
        pred_df = pd.DataFrame({"true": y_true, "pred": y_pred})
        for i, c in enumerate(cls_names):
            pred_df[f"p_{c}"] = probs[:, i].round(5)
        pred_df.to_csv(os.path.join(out, "predictions.csv"), index_label="sample")

        rows.append({"split": split, **m})
        print(f"[{split:5s}] n={m['n']:5d}  acc={100 * m['accuracy']:.2f}  "
              f"F1={100 * m['f1_macro']:.2f}  P={100 * m['precision_macro']:.2f}  "
              f"R={100 * m['recall_macro']:.2f}")

    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "summary_splits.csv"), index=False)
    model.train(was_training)
    return rows


# ============================================================
# Complete Evaluation
# ============================================================

def evaluate_model(
    history,
    y_true,
    y_pred,
    save_dir,
    dataset,
    model,
    best_epoch,
    train_size,
    test_size,
    training_time,
    class_names=None,
    total_params=None,
    trainable_params=None,
    complexity=None,
    val_size=None,
    best_val_acc=None,
    epochs_trained=None,
):
    """
    Save all evaluation results.

    Output
    ------
    history.csv                  (đã lưu ở train.py)
    metrics.json
    predictions.csv
    classification_report.csv
    confusion_matrix.csv
    confusion_matrix.png
    learning_curve.png
    """

    os.makedirs(
        save_dir,
        exist_ok=True,
    )
    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    metrics, cm, report = compute_metrics(
        y_true,
        y_pred,
    )
    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------
    save_metrics(
        save_dir=save_dir,
        metrics=metrics,
        dataset=dataset,
        model=model,
        best_epoch=best_epoch,
        train_size=train_size,
        test_size=test_size,
        training_time=training_time,
        total_params=total_params,
        trainable_params=trainable_params,
        complexity=complexity,
        val_size=val_size,
        best_val_acc=best_val_acc,
        epochs_trained=epochs_trained,
    )

    save_predictions(
        save_dir,
        y_true,
        y_pred,
    )

    save_classification_report(
        save_dir,
        report,
    )

    save_confusion_matrix(
        save_dir,
        cm,
        class_names,
    )

    plot_confusion_matrix(
        save_dir,
        cm,
        class_names,
    )

    plot_learning_curve(
        history,
        save_dir,
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Evaluation Result")
    print("=" * 60)
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Macro Precision   : {metrics['precision_macro']:.4f}")
    print(f"Macro Recall      : {metrics['recall_macro']:.4f}")
    print(f"Macro F1-score    : {metrics['f1_macro']:.4f}")
    print("=" * 60)
    return metrics