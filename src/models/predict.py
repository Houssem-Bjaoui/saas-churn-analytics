"""
predict.py
----------
Single source of truth for scoring ONE new customer record end-to-end:
raw input -> feature engineering -> encoding -> scaling -> prediction.

Critical design principle: this function must reproduce EXACTLY the same
transformation chain used in 06_modeling.ipynb (target-encoded categoricals
for SubscriptionType/PaymentMethod/GenrePreference, one-hot for the
remaining nominal categoricals, StandardScaler on numeric features).
Any drift between this function and the notebook's preprocessing would
silently produce wrong predictions in production — this is the #1 risk
in ML deployment (train/serve skew).
"""

import pandas as pd
import numpy as np

from src.features.build_features import apply_feature_engineering

# Must match src/models/train.py exactly — the same constants used to
# build X_train in 06_modeling.ipynb.
REMAINING_NOMINAL_CATEGORICALS = ["ContentType", "Gender", "ParentalControl", "SubtitlesEnabled"]
REDUNDANT_FEATURES = ["tenure_bucket", "avg_monthly_spend_historical"]
RAW_CATEGORICALS_REPLACED_BY_TARGET_ENCODING = ["SubscriptionType", "PaymentMethod", "GenrePreference"]

RISK_THRESHOLDS = {"low": 0.30, "medium": 0.60}

DEFAULT_ENGAGEMENT_COLS = [
    "ViewingHoursPerWeek",
    "AverageViewingDuration",
    "ContentDownloadsPerMonth",
]


def assign_risk_segment(probability: float) -> str:
    """Maps a churn probability to a business-readable risk tier."""
    if probability < RISK_THRESHOLDS["low"]:
        return "Low Risk"
    elif probability < RISK_THRESHOLDS["medium"]:
        return "Medium Risk"
    return "High Risk"


def _normalize_feature_stats(feature_stats: dict, df_raw: pd.DataFrame) -> dict:
    """
    Backward-compatible normalization for persisted feature stats.

    Older artifacts may miss keys introduced later (e.g.
    engagement_median, support_tickets_q75). This function backfills
    missing values with safe defaults to keep inference online.
    """
    stats = dict(feature_stats)

    engagement_cols = stats.get("engagement_cols", DEFAULT_ENGAGEMENT_COLS)
    stats["engagement_cols"] = engagement_cols

    if "engagement_mean" not in stats:
        stats["engagement_mean"] = {
            col: float(df_raw[col].mean()) if col in df_raw.columns else 0.0
            for col in engagement_cols
        }

    if "engagement_std" not in stats:
        stats["engagement_std"] = {
            col: 1.0
            for col in engagement_cols
        }
    else:
        # Avoid division-by-zero in z-score calculation.
        stats["engagement_std"] = {
            col: (1.0 if stats["engagement_std"].get(col, 0) in [0, None] else stats["engagement_std"].get(col))
            for col in engagement_cols
        }

    # engagement_score is a z-score average; threshold 0 is a reasonable
    # fallback split when train median is not persisted.
    stats.setdefault("engagement_median", 0.0)

    if "monthly_charges_q75" not in stats:
        stats["monthly_charges_q75"] = float(df_raw["MonthlyCharges"].quantile(0.75))

    if "support_tickets_q75" not in stats:
        stats["support_tickets_q75"] = float(df_raw["SupportTicketsPerMonth"].quantile(0.75))

    stats.setdefault("global_churn_rate", 0.5)

    for col in ["SubscriptionType", "PaymentMethod", "GenrePreference"]:
        stats.setdefault(f"{col}_target_encoding", {})

    return stats


def prepare_single_customer(raw_input: dict, feature_stats: dict) -> pd.DataFrame:
    """
    Transform ONE raw customer record (as received by the API) into the
    exact feature matrix format the model expects.

    Steps mirror 05_feature_engineering.ipynb + 06_modeling.ipynb, in order:
    1. Feature engineering (engagement_score, tenure flags, target encoding...)
    2. Drop raw categoricals already captured by target encoding + redundant features
    3. One-hot encode the remaining nominal categoricals
    4. Column alignment happens separately in score_customer() using the
       model's known feature list, since a single row can't reproduce
       every one-hot column that existed in the full training set.
    """
    df = pd.DataFrame([raw_input])

    normalized_stats = _normalize_feature_stats(feature_stats, df)
    df = apply_feature_engineering(df, normalized_stats)

    cols_to_drop = RAW_CATEGORICALS_REPLACED_BY_TARGET_ENCODING + REDUNDANT_FEATURES
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df = pd.get_dummies(df, columns=[c for c in REMAINING_NOMINAL_CATEGORICALS if c in df.columns])

    return df


def score_customer(raw_input: dict, model, scaler, feature_stats: dict) -> dict:
    """
    Full end-to-end scoring pipeline for a single customer.

    Returns a dict ready to be serialized as an API response —
    NOT a DataFrame, to keep the API layer simple and JSON-friendly.
    """
    df = prepare_single_customer(raw_input, feature_stats)

    # Align columns to exactly what the model was trained on. Any column
    # the model expects but this single row doesn't have (e.g. a rare
    # one-hot category absent from this one customer) is filled with 0 —
    # any unexpected extra column is silently dropped by the reindex.
    expected_features = model.get_booster().feature_names
    df = df.reindex(columns=expected_features, fill_value=0)

    # Scale numeric features using the SAME fitted scaler from Step 6 —
    # never re-fit here, only transform.
    scaled_cols = [c for c in scaler.feature_names_in_.tolist() if c in df.columns]
    df[scaled_cols] = scaler.transform(df[scaled_cols])

    probability = float(model.predict_proba(df)[0, 1])

    return {
        "churn_probability": round(probability, 4),
        "risk_segment": assign_risk_segment(probability),
    }