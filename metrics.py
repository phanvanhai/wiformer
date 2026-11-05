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
        "train_size":
            int(train_size),
        "test_size":
            int(test_size),
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
        history["test_loss"],
        label="Test",
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
        history["test_acc"],
        label="Test",
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