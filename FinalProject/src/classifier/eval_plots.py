from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _precision_recall_f1_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    support = cm.sum(axis=1).astype(np.int64)
    return precision, recall, f1, support


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path, normalize: bool):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
        cm_to_plot = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)
        title = "Confusion Matrix (normalized by true class)"
    else:
        cm_to_plot = cm
        title = "Confusion Matrix (counts)"

    plt.figure(figsize=(12, 10))
    plt.imshow(cm_to_plot, aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    plt.yticks(range(len(class_names)), class_names, fontsize=7)
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_metric_bars(values: np.ndarray, class_names: List[str], title: str, out_path: Path):
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(class_names)), values)
    plt.title(title)
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_top_confusions(cm: np.ndarray, class_names: List[str], out_dir: Path, top_k: int = 5):
    per_class_dir = _ensure_dir(out_dir / "per_class_top_confusions")

    for i, name in enumerate(class_names):
        row = cm[i].copy()
        row[i] = 0
        total = row.sum()

        if total == 0:
            continue

        top_idx = np.argsort(row)[::-1][:top_k]
        top_vals = row[top_idx]

        labels = [class_names[j] for j in top_idx]
        vals = top_vals / total

        plt.figure(figsize=(10, 4))
        plt.bar(labels, vals)
        plt.title(f"Top confusions for TRUE '{name}' (relative among mistakes)")
        plt.ylabel("Fraction of mistakes")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(per_class_dir / f"top_confusions__{name}.png", dpi=200)
        plt.close()


def save_classifier_eval_plots(
    *,
    cfg,
    predictions: Sequence[int],
    labels: Sequence[int],
    class_names: Sequence[str],
    out_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Saves evaluation plots for multi-class classification:
      - confusion_matrix_counts.png
      - confusion_matrix_normalized.png
      - per_class_precision.png
      - per_class_recall.png
      - per_class_f1.png
      - per_class_top_confusions/*.png

    Returns a dict of created key -> file path.
    """
    y_pred = np.asarray(list(predictions), dtype=np.int64)
    y_true = np.asarray(list(labels), dtype=np.int64)

    num_classes = len(class_names)
    class_names = list(class_names)

    base_results = Path(out_dir) if out_dir is not None else Path(cfg.paths.results)
    plots_dir = _ensure_dir(base_results / "classifier_eval_plots")

    cm = _confusion_matrix(y_true, y_pred, num_classes)
    precision, recall, f1, support = _precision_recall_f1_from_cm(cm)

    created = {}

    p1 = plots_dir / "confusion_matrix_counts.png"
    _plot_confusion_matrix(cm, class_names, p1, normalize=False)
    created["confusion_matrix_counts"] = str(p1)

    p2 = plots_dir / "confusion_matrix_normalized.png"
    _plot_confusion_matrix(cm, class_names, p2, normalize=True)
    created["confusion_matrix_normalized"] = str(p2)

    p3 = plots_dir / "per_class_precision.png"
    _plot_metric_bars(precision, class_names, "Per-class Precision", p3)
    created["per_class_precision"] = str(p3)

    p4 = plots_dir / "per_class_recall.png"
    _plot_metric_bars(recall, class_names, "Per-class Recall", p4)
    created["per_class_recall"] = str(p4)

    p5 = plots_dir / "per_class_f1.png"
    _plot_metric_bars(f1, class_names, "Per-class F1", p5)
    created["per_class_f1"] = str(p5)

    _plot_top_confusions(cm, class_names, plots_dir, top_k=5)
    created["per_class_top_confusions_dir"] = str(plots_dir / "per_class_top_confusions")

    return created
