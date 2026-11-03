"""
Evaluate a trained model with medical-grade metrics:
Accuracy, F1, Precision, Recall, ROC-AUC, Confusion Matrix, Quadratic Weighted Kappa.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, cohen_kappa_score, ConfusionMatrixDisplay
)


def evaluate_predictions(y_true, y_pred_probs, class_names=None):
    """
    y_true: 1D array of integer class labels
    y_pred_probs: 2D array (n_samples, n_classes) of predicted probabilities
    """
    y_pred = np.argmax(y_pred_probs, axis=1)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(y_true, y_pred_probs, multi_class="ovr"),
        "quadratic_weighted_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - DR Severity Grading")
    plt.tight_layout()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    return results, cm


if __name__ == "__main__":
    # quick smoke test with random data
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    y_probs = rng.dirichlet(np.ones(5), size=200)
    results, cm = evaluate_predictions(y_true, y_probs, class_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"])
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
