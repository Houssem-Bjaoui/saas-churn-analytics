"""
preprocessing_pipeline.py
--------------------------
Builds the reusable sklearn preprocessing pipeline used across training,
validation, test, and future production/API inference.

Design principle: this function is the SINGLE source of truth for how
raw features are transformed. It is never duplicated inline in notebooks
or in the API — everything imports from here, guaranteeing identical
preprocessing everywhere the model is used.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# Feature groups — defined once, imported wherever needed
NUMERIC_FEATURES = [
    "AccountAge", "MonthlyCharges", "TotalCharges", "ViewingHoursPerWeek",
    "AverageViewingDuration", "ContentDownloadsPerMonth", "UserRating",
    "SupportTicketsPerMonth", "WatchlistSize"
]

ORDINAL_FEATURES = ["SubscriptionType"]
ORDINAL_CATEGORIES = [["Basic", "Standard", "Premium"]]

NOMINAL_FEATURES = [
    "PaymentMethod", "ContentType", "GenrePreference",
    "Gender", "ParentalControl", "SubtitlesEnabled"
]


def build_preprocessing_pipeline(
    numeric_features: list = NUMERIC_FEATURES,
    ordinal_features: list = ORDINAL_FEATURES,
    ordinal_categories: list = ORDINAL_CATEGORIES,
    nominal_features: list = NOMINAL_FEATURES,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer applying the right transformation per feature type.

    - Numeric: median imputation + standard scaling.
    - Ordinal (SubscriptionType): preserves Basic < Standard < Premium order,
      justified by its monotonic relationship with churn found in EDA.
    - Nominal: One-Hot encoding, no assumed order.

    handle_unknown settings protect the pipeline against unseen categories
    appearing later in test/production data without crashing.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    nominal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("ord", ordinal_transformer, ordinal_features),
        ("nom", nominal_transformer, nominal_features),
    ])

    return preprocessor