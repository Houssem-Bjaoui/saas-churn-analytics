"""
schemas.py
----------
Pydantic request/response models for the churn prediction API.
Field names match the raw dataset columns exactly (see docs/data_dictionary.md)
so no renaming/mapping is needed between the API layer and the feature
engineering pipeline.
"""

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    AccountAge: int = Field(..., ge=0, description="Account age in months")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    SubscriptionType: str = Field(..., description="Basic, Standard, or Premium")
    PaymentMethod: str
    ContentType: str
    ViewingHoursPerWeek: float = Field(..., ge=0)
    AverageViewingDuration: float = Field(..., ge=0)
    ContentDownloadsPerMonth: int = Field(..., ge=0)
    GenrePreference: str
    UserRating: float = Field(..., ge=1, le=5)
    SupportTicketsPerMonth: int = Field(..., ge=0)
    Gender: str
    WatchlistSize: int = Field(..., ge=0)
    ParentalControl: str = Field(..., description="Yes or No")
    SubtitlesEnabled: str = Field(..., description="Yes or No")

    class Config:
        json_schema_extra = {
            "example": {
                "AccountAge": 5,
                "MonthlyCharges": 18.5,
                "TotalCharges": 92.5,
                "SubscriptionType": "Basic",
                "PaymentMethod": "Electronic check",
                "ContentType": "Both",
                "ViewingHoursPerWeek": 4.2,
                "AverageViewingDuration": 22.0,
                "ContentDownloadsPerMonth": 1,
                "GenrePreference": "Comedy",
                "UserRating": 2.8,
                "SupportTicketsPerMonth": 6,
                "Gender": "Male",
                "WatchlistSize": 2,
                "ParentalControl": "No",
                "SubtitlesEnabled": "No"
            }
        }


class ChurnPrediction(BaseModel):
    churn_probability: float
    risk_segment: str