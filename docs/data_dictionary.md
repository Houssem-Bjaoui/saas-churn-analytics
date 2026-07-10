# Data Dictionary: SaaS Churn Prediction

This document lists and explains the variables available in the SaaS subscription dataset. Understanding these attributes is crucial to identifying early warning signals of churn.

| Variable | Type | Description | Business Hypothesis (SaaS Context) |
| :--- | :--- | :--- | :--- |
| **CustomerID** | `String` | Unique customer identifier. | *Should be excluded from modeling (noise).* |
| **SubscriptionType** | `Categorical` | Selected plan (e.g., Basic, Premium, Deluxe). | "Basic" plans may have higher churn (lower engagement), or the opposite (Premium too expensive). |
| **PaymentMethod** | `Categorical` | Payment method (e.g., Credit Card, PayPal). | Manual payments (Check) have historically shown higher failure/passive churn rates than auto-pay. |
| **PaperlessBilling** | `Binary` | Paperless billing enabled (Yes/No). | Linked to customer digital adoption. |
| **ContentType** | `Categorical` | Type of content consumed (e.g., Movies, TV Shows). | Reveals the customer's primary use case. |
| **MultiDeviceAccess** | `Binary` | Multi-device access enabled (Yes/No). | A multi-device customer is generally more embedded in the product ecosystem (stronger retention). |
| **DeviceRegistered** | `Categorical` | Primary device (Smartphone, Smart TV...). | Impacts user experience (UX). |
| **GenrePreference** | `Categorical` | Preferred content genre. | Enables marketing segmentation analyses. |
| **Gender** | `Categorical` | Customer gender. | Historically low predictive power for SaaS churn, but should be checked for model fairness. |
| **ParentalControl** | `Binary` | Parental control enabled (Yes/No). | Indicator of children in the household. Family households often show stronger cancellation inertia. |
| **SubtitlesEnabled** | `Binary` | Subtitles enabled (Yes/No). | Behavioral/accessibility indicator. |
| **AccountAge** | `Numeric` | Account age in months. | **Critical.** Churn risk is generally very high in the first 3 months, then stabilizes (survival curve). |
| **MonthlyCharges** | `Numeric` | Monthly bill amount. | Price sensitivity. Sharp increases often drive churn. |
| **TotalCharges** | `Numeric` | Total amount billed since sign-up. | Highly correlated with `AccountAge` and `MonthlyCharges`. Watch for collinearity. |
| **ViewingHoursPerWeek**| `Numeric` | Weekly viewing hours. | Direct engagement indicator (Health Score). A sudden drop often precedes churn. |
| **SupportTicketsPerMonth**| `Numeric`| Number of support tickets per month. | **Friction signal.** Too many tickets = frustration. Zero tickets with low usage = "zombie" customer (high risk). |
| **AverageViewingDuration**| `Numeric`| Average session duration. | Engagement quality (binge-watching vs. micro-sessions). |
| **ContentDownloadsPerMonth**|`Numeric`| Number of downloads. | Indicates offline usage, a sign of strong product adoption. |
| **UserRating** | `Numeric` | Satisfaction rating (1 to 5). | Explicit NPS/CSAT signal. Ratings < 3 are obvious red flags. |
| **WatchlistSize** | `Numeric` | Size of the "Watchlist". | Future intent indicator. An empty list suggests low long-term interest. |
| **Churn** *(Target)* | `Binary` | Has the customer churned? (Yes/No / 1/0) | **Target variable to predict.** |