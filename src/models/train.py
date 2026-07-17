"""
train.py
--------
Model building and training utilities for the churn prediction pipeline.

Design principle: model definitions and hyperparameter grids live here,
so the notebook only orchestrates (calls functions, compares results,
interprets) rather than duplicating model configuration inline.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def get_baseline_models(y_train) -> dict:
    """
    Return a dictionary of baseline models, each configured to handle
    class imbalance natively via class weighting rather than resampling.

    Rationale (documented in Step 4): churn rate is ~18%, a moderate
    (not extreme) imbalance. class_weight="balanced" is preferred over
    SMOTE as the default strategy because it doesn't alter the data
    distribution and is easier to justify/reproduce. scale_pos_weight
    for XGBoost achieves the equivalent effect.
    """
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    scale_pos_weight = neg / pos

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
    }
    return models


def get_xgboost_param_grid() -> dict:
    """
    Hyperparameter search space for XGBoost tuning.
    Kept intentionally moderate in size (3x3x3x2 = 54 combinations x 5 folds
    = 270 fits) to stay tractable without a distributed compute cluster —
    appropriate for a portfolio project, not a production-scale search.
    """
    return {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 300, 500],
        "subsample": [0.8, 1.0],
    }


def tune_xgboost(X_train, y_train, scale_pos_weight: float, cv_folds: int = 5) -> GridSearchCV:
    """
    Run a stratified cross-validated grid search for XGBoost, optimizing
    for ROC-AUC. Stratified folds preserve the ~18% churn ratio in every
    fold, consistent with the train/val split discipline from Step 4.
    """
    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=get_xgboost_param_grid(),
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search