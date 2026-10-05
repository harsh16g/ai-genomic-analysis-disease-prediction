"""
visualize.py
--------------
Plotting helpers for the report — satisfies the syllabus's "Biological
data visualization" session (CO3) and gives the automation pipeline
something to save as image artifacts.

Usage (after running pipeline.py, or interactively):

    from visualize import plot_confusion_matrix, plot_gc_distribution
    plot_confusion_matrix(cm, class_names, "../outputs/rf_confusion_matrix.png")
"""

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(cm, class_names, out_path: str, title: str = "Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_gc_distribution(sequences: list, labels: list, out_path: str):
    """Boxplot of GC content per class — a quick visual sanity check that
    classes are separable before training."""
    df = pd.DataFrame({
        "gc_content": [s.gc_content() for s in sequences],
        "label": labels,
    })
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="label", y="gc_content")
    plt.xticks(rotation=30, ha="right")
    plt.title("GC Content Distribution by Class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_model_accuracy_comparison(results: dict, out_path: str):
    """Bar chart comparing test accuracy across classical ML models
    (and CNN, if its result dict is merged in under key 'CNN')."""
    names, accs = [], []
    for name, r in results.items():
        if name in ("label_encoder", "scaler"):
            continue
        names.append(name)
        accs.append(r["test_accuracy"])

    plt.figure(figsize=(6, 4))
    sns.barplot(x=names, y=accs)
    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
