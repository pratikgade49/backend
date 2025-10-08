"""
Pydantic schemas for request/response validation
"""

from app.schemas.auth import *
from app.schemas.user import *
from app.schemas.forecast import *
from app.schemas.configuration import *
from app.schemas.external_factor import *
from app.schemas.scheduler import *
from app.schemas.stats import *
from app.schemas.saved_forecast import *


__all__ = [
    "UserCreate", "UserLogin", "UserResponse", "AdminSetActiveRequest",
    "Token",
    "ForecastConfig", "DataPoint", "AlgorithmResult", "ForecastResult",
    "MultiForecastResult", "SavedForecastRequest", "SavedForecastResponse",
    "BestFitRecommendationRequest", "BestFitRecommendationResponse",
    "SaveConfigRequest", "ConfigurationResponse",
    "FredDataRequest", "FredDataResponse",
    "ScheduledForecastCreate", "ScheduledForecastUpdate",
    "ScheduledForecastResponse", "ForecastExecutionResponse",
    "DatabaseStats", "DataViewRequest", "DataViewResponse"
]
