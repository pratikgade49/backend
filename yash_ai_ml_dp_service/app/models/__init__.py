"""
Database models
"""
from app.core.database import Base
from app.models.user import User
from app.models.roles import UserRole
from app.models.dimension import (
    ProductDimension,
    CustomerDimension,
    LocationDimension,
    ProductCustomerCombination,
    ProductLocationCombination,
    CustomerLocationCombination,
    ProductCustomerLocationCombination
)
from app.models.forecast import (
    ForecastData,
    ForecastConfiguration
)
from app.models.saved_forecast import SavedForecastResult
from app.models.external_factor import ExternalFactorData
from app.models.scheduler import (
    ScheduledForecast,
    ForecastExecution,
    ScheduleFrequency,
    ScheduleStatus
)
from app.models.model_persistence import (
    SavedModel,
    ModelAccuracyHistory
)

__all__ = [
    "Base",
    "User",
    "UserRole",
    "ProductDimension",
    "CustomerDimension",
    "LocationDimension",
    "ProductCustomerCombination",
    "ProductLocationCombination",
    "CustomerLocationCombination",
    "ProductCustomerLocationCombination",
    "ForecastData",
    "SavedForecastResult",
    "ForecastConfiguration",
    "ExternalFactorData",
    "ScheduledForecast",
    "ForecastExecution",
    "ScheduleFrequency",
    "ScheduleStatus",
    "SavedModel",
    "ModelAccuracyHistory"
]
