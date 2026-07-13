"""
clean_data.py
-------------
Data quality checks and column selection logic, applied before the
preprocessing pipeline. Decisions here are justified by the EDA findings
documented in 02_eda.ipynb (Chi-square tests, correlation analysis).
"""

import pandas as pd

ID_COL = "CustomerID"
TARGET = "Churn"

# Excluded: no statistically significant relationship with churn
# (Chi-square p-values 0.42, 0.53, 0.65 respectively — see EDA notebook).
EXCLUDED_FEATURES = ["DeviceRegistered", "MultiDeviceAccess", "PaperlessBilling"]


def run_pre_cleaning_checks(df: pd.DataFrame, name: str) -> dict:
    """
    Re-verify data quality assumptions before cleaning. Returns a dict
    instead of only printing, so results can be asserted in tests or
    logged in a pipeline run.
    """
    results = {
        "name": name,
        "avg_missing_pct": df.isna().mean().mean() * 100,
        "n_duplicate_rows": int(df.duplicated().sum()),
    }
    if ID_COL in df.columns:
        results["n_duplicated_ids"] = int(df.duplicated(subset=[ID_COL]).sum())

    return results


def select_modeling_columns(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """
    Build the working dataframe used for modeling:
    - Drops the identifier column (not predictive).
    - Drops low-signal columns identified as statistically insignificant in EDA.
    - Keeps the target column only when include_target=True (test.csv has
      no Churn column — confirmed in Step 1 as a genuine unseen prediction set).
    """
    cols_to_drop = [ID_COL] + EXCLUDED_FEATURES
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    df_clean = df.drop(columns=cols_to_drop)

    if not include_target and TARGET in df_clean.columns:
        df_clean = df_clean.drop(columns=[TARGET])

    return df_clean


def inspect_total_charges_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspect (not remove) TotalCharges outliers using the IQR method.
    Returns the outlier subset with AccountAge/Churn context, so the
    "legitimate long-tenure customer" hypothesis from the EDA can be
    verified programmatically rather than assumed.
    """
    q1, q3 = df["TotalCharges"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = df["TotalCharges"] > upper_bound
    return df.loc[outlier_mask, ["AccountAge", "MonthlyCharges", "TotalCharges", TARGET]]