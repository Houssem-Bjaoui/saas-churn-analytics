"""
shap_analysis.py
-----------------
SHAP-based explainability utilities for the final XGBoost churn model.

Design principle: this module is intentionally independent of the
training logic (src/models/train.py) — it takes an already-fitted model
and transformed features as input. This keeps explainability decoupled
from training, so it can be re-run on any saved model without retraining.
"""

import shap
import pandas as pd
import numpy as np


def build_shap_explainer(model):
    """
    Build a SHAP TreeExplainer for the final XGBoost model.
    TreeExplainer is used (rather than the generic/model-agnostic
    KernelExplainer) because it is exact and fast for tree-based models —
    no sampling approximation needed, unlike KernelExplainer which would
    be required for Logistic Regression or a black-box model.
    """
    return shap.TreeExplainer(model)


def compute_shap_values(explainer, X: pd.DataFrame):
    """
    Compute SHAP values for a given feature matrix.
    Returns an array of shape (n_samples, n_features), where each value
    represents how much that feature pushed a specific prediction away
    from the model's baseline (expected) output.
    """
    return explainer.shap_values(X)


def get_global_feature_ranking(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Rank features by their mean absolute SHAP value — the standard way
    to summarize global feature importance from SHAP, since positive and
    negative contributions would otherwise cancel out in a plain average.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return ranking


def explain_single_customer(explainer, shap_values, X: pd.DataFrame, row_index: int, top_n: int = 5) -> pd.DataFrame:
    """
    Build a readable, ranked table of the top contributing features for
    ONE specific customer's prediction — the output a Customer Success
    team would actually use to understand "why is this customer flagged?"

    Positive SHAP value = pushes prediction toward churn (higher risk).
    Negative SHAP value = pushes prediction toward retention (lower risk).
    """
    customer_shap = shap_values[row_index]
    customer_features = X.iloc[row_index]

    explanation = pd.DataFrame({
        "feature": X.columns,
        "feature_value": customer_features.values,
        "shap_value": customer_shap
    })

    explanation["abs_shap"] = explanation["shap_value"].abs()
    explanation = explanation.sort_values("abs_shap", ascending=False).head(top_n)
    explanation["direction"] = np.where(
        explanation["shap_value"] > 0, "-> increases churn risk", "-> decreases churn risk"
    )

    return explanation.drop(columns="abs_shap").reset_index(drop=True)