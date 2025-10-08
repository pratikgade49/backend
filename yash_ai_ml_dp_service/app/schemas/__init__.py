"""
Pydantic schemas for request/response validation
"""

from app.schemas.user import UserCreate, UserLogin, UserResponse, AdminSetActiveRequest
from app.schemas.auth import Token
from app.schemas.forecast import (
    ForecastConfig,
    DataPoint,
    AlgorithmResult,
    ForecastResult,
    MultiForecastResult,
    SavedForecastRequest,
    SavedForecastResponse,
    BestFitRecommendationRequest,
    BestFitRecommendationResponse
)
from app.schemas.configuration import SaveConfigRequest, ConfigurationResponse
from app.schemas.external_factor import FredDataRequest, FredDataResponse
from app.schemas.scheduler import (
    ScheduledForecastCreate,
    ScheduledForecastUpdate,
    ScheduledForecastResponse,
    ForecastExecutionResponse
)
from app.schemas.stats import DatabaseStats, DataViewRequest, DataViewResponse

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
