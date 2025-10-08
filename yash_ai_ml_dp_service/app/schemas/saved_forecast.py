#!/usr/bin/env python3
"""
Saved forecast schemas
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class SavedForecastRequest(BaseModel):
    """Schema for saving forecast result"""
    name: str
    config: dict
    forecast_data: List[Dict[str, Any]]
    historic_data: List[Dict[str, Any]]
    algorithm: str
    accuracy: float

class SavedForecastResponse(BaseModel):
    """Schema for saved forecast response"""
    id: int
    user_id: int
    name: str
    config: dict
    algorithm: str
    accuracy: float
    created_at: str

    class Config:
        from_attributes = True
