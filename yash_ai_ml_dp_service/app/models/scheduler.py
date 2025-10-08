#!/usr/bin/env python3
"""
Scheduler models for automated forecast generation
"""

from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from enum import Enum as PyEnum
from app.core.database import Base

class ScheduleFrequency(PyEnum):
    """Schedule frequency enumeration"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class ScheduleStatus(PyEnum):
    """Schedule status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class ScheduledForecast(Base):
    """Model for storing scheduled forecast configurations"""
    __tablename__ = "scheduled_forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    forecast_config = Column(Text, nullable=False)
    frequency = Column(String(20), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)
    next_run = Column(DateTime, nullable=False)
    last_run = Column(DateTime, nullable=True)
    status = Column(String(20), default='active')
    run_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ForecastExecution(Base):
    """Model for storing forecast execution history"""
    __tablename__ = "forecast_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    scheduled_forecast_id = Column(Integer, nullable=False, index=True)
    execution_time = Column(DateTime, nullable=False)
    status = Column(String(50), nullable=False)
    duration_seconds = Column(Integer, nullable=True)
    result_summary = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    forecast_data = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
