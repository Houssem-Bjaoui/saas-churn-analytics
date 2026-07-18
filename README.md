# SaaS Subscription Churn Prediction 
An end-to-end churn analytics project for a subscription-based streaming/media service. The goal is to identify customers at risk of leaving, explain the main drivers of churn, and support retention decisions with a reproducible ML pipeline.

## Business Problem
Subscription services live or die by retention. Even a relatively small churn rate compounds into large monthly revenue loss, so the challenge is not only to predict churn, but to detect it early enough to act.

This project focuses on understanding which customer behaviors and plan attributes are most associated with churn, then turning those signals into an operational prediction workflow that can support customer success and retention teams.

## Dataset
Source: Kaggle subscription churn dataset for a streaming/media service.

- Size: 243,787 rows and 21 columns
- Target: `Churn`
- Data quality: 0 missing values, 0 duplicate rows in the training snapshot
- Domain context: subscription-based streaming/media service

The dataset contains a mix of behavioral, usage, billing, and plan-related variables such as `AccountAge`, `MonthlyCharges`, `TotalCharges`, `ViewingHoursPerWeek`, `SupportTicketsPerMonth`, and several categorical plan/engagement fields.

## Project Architecture
The project is organized as a notebook-driven analysis with reusable source modules.

- `notebooks/01_data_understanding.ipynb`: initial inspection and data quality checks
- `notebooks/02_eda.ipynb`: exploratory analysis, correlation analysis, and statistical testing
- `notebooks/03_cleaning.ipynb`: column selection and cleaning rules
- `notebooks/04_saas_metrics.ipynb`: business metrics such as MRR, ARPU, CLV, and churn loss
- `notebooks/05_feature_engineering.ipynb`: engineered retention features and target encoding
- `notebooks/06_modeling.ipynb`: preprocessing, model training, tuning, and evaluation
- `notebooks/07_explainability.ipynb`: SHAP analysis and business-readable feature ranking

Reusable code lives in `src/`:

- `src/data/clean_data.py`: data validation and modeling column selection
- `src/features/build_features.py`: feature engineering and target encoding
- `src/pipeline/preprocessing_pipeline.py`: sklearn preprocessing pipeline
- `src/models/train.py`: baseline models and XGBoost tuning
- `src/models/evaluate.py`: metric computation and plots
- `src/models/predict.py`: end-to-end scoring for one customer
- `src/explainability/shap_analysis.py`: SHAP explainability helpers

Artifacts are stored in:

- `models/`: trained model, scaler, SHAP explainer, and feature statistics
- `data/processed/`: engineered datasets, business metrics, and SHAP ranking outputs
- `reports/figures/`: generated charts and explainability plots
- `api/`: FastAPI service exposing the scoring endpoint

## Key Findings
- The dataset is structurally clean: 243,787 rows, no missing values, and no duplicates.
- Overall churn rate is 18.12%, which is high enough to justify a recall-focused modeling strategy.
- `AccountAge` is the strongest linear churn driver: longer tenure is associated with lower churn.
- `AccountAge` and `TotalCharges` are strongly correlated ($r \approx 0.82$), which justified a ratio-style feature to reduce redundancy.
- `MonthlyCharges` is positively associated with churn, and `SupportTicketsPerMonth` also shows a clear positive relationship with churn.
- Categorical variables such as `SubscriptionType`, `PaymentMethod`, and `GenrePreference` are statistically significant churn drivers.
- Behavioral engagement matters: usage and engagement signals outperform many raw plan attributes once combined into stronger composite features.

## Model Performance
- Final model: XGBoost (tuned)
- ROC-AUC: 0.7532
- Recall: 0.7009
- Precision: 0.3196
- F1: 0.4390

Selection rationale: the project prioritizes recall because missing a churner is more expensive than flagging an extra at-risk customer. Tuned XGBoost was selected because it provided the best balance of ranking quality and churn capture among the tested models.

## Explainability
SHAP confirmed the main EDA conclusions and produced a business-readable ranking of the most important drivers.

Top global features by mean absolute SHAP:

1. `engagement_score`
2. `AccountAge`
3. `tickets_per_tenure_month`
4. `MonthlyCharges`
5. `SubscriptionType_risk_encoded`

This is consistent with the EDA: engagement and tenure dominate, while pricing and support friction act as secondary risk signals. The SHAP outputs also support the feature engineering choices made earlier in the project.

## How to Run
### 1. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the API locally
```bash
uvicorn api.main:app --reload
```

Open the interactive docs at:
- `http://127.0.0.1:8000/docs`

### 3. Run with Docker
The project also includes an API `Dockerfile`.

```bash
docker build -t saas-churn-api -f api/Dockerfile .
docker run -p 8000:8000 saas-churn-api
```

### 4. Reproduce the notebooks
Open the notebooks in order from `notebooks/01_data_understanding.ipynb` to `notebooks/07_explainability.ipynb`.

## Tech Stack
- Python 3.11
- pandas, numpy
- scikit-learn
- XGBoost
- SHAP
- matplotlib, seaborn
- FastAPI, Pydantic, Uvicorn
- joblib
- Jupyter Notebook

## Project Status
Completed end-to-end project with:

- cleaned and analyzed data
- engineered features
- trained and evaluated models
- persisted model artifacts
- explainability outputs
- a working FastAPI scoring service

## Future Work
- Calibrate predicted probabilities and tune the decision threshold for operations.
- Add automated tests around preprocessing and scoring.
- Track feature drift and prediction drift after deployment.
- Add a batch scoring pipeline for campaign targeting.
- Compare XGBoost with a simpler calibrated baseline for easier interpretability.
