"""
evaluate.py
-----------
Evaluation utilities for churn prediction models: metrics computation,
model comparison tables, and confusion matrix visualization.

Design principle: accuracy is intentionally NOT the headline metric.
Given the ~18% churn imbalance (confirmed in Step 1), Recall, Precision,
F1, and ROC-AUC are tracked as the primary metrics throughout.
"""

import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


def evaluate_model(model, X_val, y_val, model_name: str) -> dict:
    """
    Compute the full metric set for a fitted model on a validation set.
    Returns a flat dict — designed to be collected into a comparison
    DataFrame across multiple models (see compare_models below).
    """
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    return {
        "model": model_name,
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
    }


def compare_models(results: list) -> pd.DataFrame:
    """
    Build a ranked comparison table from a list of evaluate_model() outputs.
    Sorted by ROC-AUC — the metric least sensitive to the decision
    threshold, making it the fairest primary ranking criterion when
    comparing models before any threshold tuning has been done.
    """
    df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    return df.reset_index(drop=True)


def plot_confusion_matrix(model, X_val, y_val, model_name: str, save_path: str = None):
    """
    Plot the confusion matrix with business-relevant labels.
    False Negatives (bottom-left: predicted retained, actually churned)
    are the costliest error type in a churn use case — a lost customer
    the business never attempted to retain.
    """
    y_pred = model.predict(X_val)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_val, y_pred,
        display_labels=["Retained", "Churned"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def print_classification_report(model, X_val, y_val, model_name: str):
    y_pred = model.predict(X_val)
    print(f"=== {model_name} — Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=["Retained", "Churned"]))