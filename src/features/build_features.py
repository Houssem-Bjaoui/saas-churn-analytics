"""
build_features.py
------------------
Feature engineering logic, driven directly by the EDA findings:

- AccountAge is the strongest churn driver but highly correlated (0.82)
  with TotalCharges -> engineer a ratio feature to reduce redundancy.
- ViewingHoursPerWeek, AverageViewingDuration, ContentDownloadsPerMonth
  move together and in the same direction -> combine into one
  composite engagement score.
- Churners tend to be on higher MonthlyCharges but with shorter tenure
  (positive corr with MonthlyCharges, negative with TotalCharges) ->
  engineer a "price sensitivity at onboarding" flag.
- SupportTicketsPerMonth has a moderate positive relationship with churn.
- SubscriptionType, PaymentMethod, GenrePreference are the strongest
  categorical drivers (Chi-square) -> candidates for target encoding.

Design principle: statistics needed to build a feature (means, stds,
quantile cutoffs, target-encoded rates) are learned ONLY on the training
fold (fit_feature_stats) and then applied identically to validation/test
(apply_feature_engineering), to avoid data leakage — the same discipline
used for the sklearn preprocessing pipeline.
"""

import pandas as pd


def fit_feature_stats(df: pd.DataFrame, target_col: str = "Churn") -> dict:
    """
    Learn all statistics needed for feature engineering from the
    TRAINING data only. Returns a dict of stats to be reused identically
    on validation/test data via apply_feature_engineering().
    """
    stats = {}

    # --- Engagement composite: needs mean/std of each component to
    # combine them on a comparable scale (z-score average). ---
    engagement_cols = ["ViewingHoursPerWeek", "AverageViewingDuration", "ContentDownloadsPerMonth"]
    stats["engagement_cols"] = engagement_cols
    stats["engagement_mean"] = df[engagement_cols].mean().to_dict()
    stats["engagement_std"] = df[engagement_cols].std().to_dict()
    stats["engagement_median"] = df[engagement_cols].mean(axis=1).median()

    # --- Value segmentation cutoffs (quantiles) for MonthlyCharges ---
    stats["monthly_charges_q75"] = df["MonthlyCharges"].quantile(0.75)
    stats["support_tickets_q75"] = df["SupportTicketsPerMonth"].quantile(0.75)

    # --- Target encoding: average churn rate per category, learned on
    # train only. Smoothing (Bayesian-style) prevents overfitting on
    # rare categories with few samples. ---
    global_churn_rate = df[target_col].mean()
    stats["global_churn_rate"] = global_churn_rate

    for col in ["SubscriptionType", "PaymentMethod", "GenrePreference"]:
        category_stats = df.groupby(col)[target_col].agg(["mean", "count"])
        smoothing = 20  # higher = more conservative shrinkage toward global rate
        smoothed_rate = (
            (category_stats["mean"] * category_stats["count"] + global_churn_rate * smoothing)
            / (category_stats["count"] + smoothing)
        )
        stats[f"{col}_target_encoding"] = smoothed_rate.to_dict()

    return stats


def apply_feature_engineering(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Apply all engineered features to a dataframe (train, val, or test),
    using ONLY the statistics learned via fit_feature_stats() on train.
    Never recomputes stats from the input df itself.
    """
    df = df.copy()

    # ============================================================
    # 1. TENURE-BASED FEATURES
    # ============================================================
    # Business logic: AccountAge was the single strongest churn driver
    # (correlation -0.20, t-stat -103.4) with a clear non-linear cohort
    # pattern — churn drops steadily as tenure increases, with the
    # sharpest risk in the first 6-12 months (onboarding churn).

    df["tenure_bucket"] = pd.cut(
        df["AccountAge"], bins=[0, 6, 12, 24, 36, 60, 200],
        labels=["0-6m", "6-12m", "1-2y", "2-3y", "3-5y", "5y+"]
    )

    # Binary flag isolates the highest-risk cohort explicitly — easier
    # for linear models to exploit than the raw non-linear cohort curve.
    df["is_early_lifecycle"] = (df["AccountAge"] <= 12).astype(int)

    # Ratio feature to reduce the AccountAge/TotalCharges redundancy
    # (0.82 correlation found in EDA) while keeping the underlying signal:
    # this represents an "average historical monthly spend" per customer,
    # independent of raw tenure length.
    df["avg_monthly_spend_historical"] = df["TotalCharges"] / df["AccountAge"].clip(lower=1)

    # ============================================================
    # 2. ENGAGEMENT COMPOSITE SCORE
    # ============================================================
    # Business logic: ViewingHoursPerWeek, AverageViewingDuration, and
    # ContentDownloadsPerMonth all correlated with churn in the same
    # direction and at similar strength (-0.13 to -0.15) — a coherent
    # underlying "engagement" signal spread across 3 metrics rather than
    # concentrated in one. Combining them into a single z-score average
    # produces a stronger, less noisy signal than any single metric alone.

    z_scores = []
    for col in stats["engagement_cols"]:
        mean = stats["engagement_mean"][col]
        std = stats["engagement_std"][col]
        z_scores.append((df[col] - mean) / std)

    df["engagement_score"] = pd.concat(z_scores, axis=1).mean(axis=1)

    # Binary low-engagement flag, using the composite score rather than
    # a single raw metric — more robust to noise in any one dimension.
    df["is_low_engagement"] = (df["engagement_score"] < stats["engagement_median"]).astype(int)

    # ============================================================
    # 3. VALUE / PRICE-SENSITIVITY FEATURES
    # ============================================================
    # Business logic: MonthlyCharges was POSITIVELY correlated with churn
    # (+0.10) while TotalCharges was NEGATIVELY correlated (-0.12) — this
    # combination suggested churners are disproportionately customers on
    # higher-priced plans who leave BEFORE accumulating tenure/spend,
    # i.e. a price-sensitivity-at-onboarding pattern, not a long-term
    # value customer leaving. This flag isolates exactly that pattern.

    df["is_high_value_plan"] = (df["MonthlyCharges"] > stats["monthly_charges_q75"]).astype(int)
    df["is_high_price_new_customer"] = (
        (df["is_high_value_plan"] == 1) & (df["is_early_lifecycle"] == 1)
    ).astype(int)

    # ============================================================
    # 4. SUPPORT INTERACTION FEATURES
    # ============================================================
    # Business logic: SupportTicketsPerMonth showed a moderate positive
    # relationship with churn (+0.08 correlation, higher mean in churners
    # per the t-test). Normalizing by tenure distinguishes a genuinely
    # high-friction customer from one who has simply had more time to
    # accumulate tickets.

    df["tickets_per_tenure_month"] = df["SupportTicketsPerMonth"] / df["AccountAge"].clip(lower=1)
    df["is_high_support_friction"] = (
        df["SupportTicketsPerMonth"] > stats["support_tickets_q75"]
    ).astype(int)

    # ============================================================
    # 5. TARGET-ENCODED CATEGORICAL RISK FEATURES
    # ============================================================
    # Business logic: SubscriptionType, PaymentMethod, and GenrePreference
    # were the three strongest categorical churn drivers (Chi-square).
    # Target encoding converts each category directly into its historical
    # churn rate (learned on train only, smoothed to avoid overfitting on
    # rare categories) — often more predictive for tree-based models than
    # one-hot encoding alone, since it directly encodes the risk ordering
    # between categories (e.g. Basic > Standard > Premium risk).

    for col in ["SubscriptionType", "PaymentMethod", "GenrePreference"]:
        encoding_map = stats[f"{col}_target_encoding"]
        df[f"{col}_risk_encoded"] = df[col].map(encoding_map).fillna(stats["global_churn_rate"])

    # ============================================================
    # 6. COMPOSITE INTERACTION: ENGAGEMENT x SUPPORT FRICTION
    # ============================================================
    # Business logic: a customer who is BOTH disengaged AND generating
    # high support friction is a compounding risk profile, likely more
    # predictive than either signal alone. This mirrors the "combine two
    # weak signals into a stronger composite" logic already validated by
    # the engagement_score feature above.

    df["disengaged_and_high_friction"] = (
        (df["is_low_engagement"] == 1) & (df["is_high_support_friction"] == 1)
    ).astype(int)

    return df