#!/usr/bin/env python3
"""
Scheduler schemas
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ScheduledForecastCreate(BaseModel):
    """Schema for creating scheduled forecast"""
    name: str
    description: Optional[str] = None
    forecast_config: dict
    frequency: str
    start_date: datetime
    end_date: Optional[datetime] = None

class ScheduledForecastUpdate(BaseModel):
    """Schema for updating scheduled forecast"""
    name: Optional[str] = None
    description: Optional[str] = None
    frequency: Optional[str] = None
    status: Optional[str] = None

class ScheduledForecastResponse(BaseModel):
    """Schema for scheduled forecast response"""
    id: int
    user_id: int
    name: str
    description: Optional[str]
    frequency: str
    start_date: datetime
    end_date: Optional[datetime]
    next_run: datetime
    last_run: Optional[datetime]
    status: str
    run_count: int
    success_count: int
    failure_count: int
    created_at: datetime

    class Config:
        from_attributes = True

class ForecastExecutionResponse(BaseModel):
    """Schema for forecast execution response"""
    id: int
    scheduled_forecast_id: int
    execution_time: datetime
    status: str
    duration_seconds: Optional[int]
    error_message: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
