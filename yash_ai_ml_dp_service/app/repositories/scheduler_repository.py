#!/usr/bin/env python3
"""
Scheduler repository for scheduled forecasts
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime
from app.models.scheduler import ScheduledForecast, ForecastExecution

def create_scheduled_forecast(
    db: Session,
    user_id: int,
    name: str,
    description: Optional[str],
    forecast_config: str,
    frequency: str,
    start_date: datetime,
    end_date: Optional[datetime],
    next_run: datetime
) -> ScheduledForecast:
    """Create a new scheduled forecast"""
    scheduled_forecast = ScheduledForecast(
        user_id=user_id,
        name=name,
        description=description,
        forecast_config=forecast_config,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
        next_run=next_run,
        status='active'
    )
    db.add(scheduled_forecast)
    db.commit()
    db.refresh(scheduled_forecast)
    return scheduled_forecast

def get_scheduled_forecast_by_id(
    db: Session,
    schedule_id: int,
    user_id: Optional[int] = None
) -> Optional[ScheduledForecast]:
    """Get scheduled forecast by ID"""
    query = db.query(ScheduledForecast).filter(ScheduledForecast.id == schedule_id)
    if user_id:
        query = query.filter(ScheduledForecast.user_id == user_id)
    return query.first()

def get_scheduled_forecasts_by_user(db: Session, user_id: int) -> List[ScheduledForecast]:
    """Get all scheduled forecasts for a user"""
    return db.query(ScheduledForecast).filter(
        ScheduledForecast.user_id == user_id
    ).order_by(ScheduledForecast.created_at.desc()).all()

def get_all_scheduled_forecasts(db: Session) -> List[ScheduledForecast]:
    """Get all scheduled forecasts"""
    return db.query(ScheduledForecast).all()

def update_scheduled_forecast(
    db: Session,
    schedule_id: int,
    **kwargs
) -> Optional[ScheduledForecast]:
    """Update scheduled forecast"""
    scheduled_forecast = db.query(ScheduledForecast).filter(
        ScheduledForecast.id == schedule_id
    ).first()
    
    if scheduled_forecast:
        for key, value in kwargs.items():
            if hasattr(scheduled_forecast, key) and value is not None:
                setattr(scheduled_forecast, key, value)
        db.commit()
        db.refresh(scheduled_forecast)
    
    return scheduled_forecast

def delete_scheduled_forecast(db: Session, schedule_id: int, user_id: int) -> bool:
    """Delete scheduled forecast"""
    scheduled_forecast = db.query(ScheduledForecast).filter(
        ScheduledForecast.id == schedule_id,
        ScheduledForecast.user_id == user_id
    ).first()
    
    if scheduled_forecast:
        db.delete(scheduled_forecast)
        db.commit()
        return True
    return False

def get_due_forecasts(db: Session) -> List[ScheduledForecast]:
    """Get forecasts that are due to run"""
    now = datetime.utcnow()
    return db.query(ScheduledForecast).filter(
        ScheduledForecast.status == 'active',
        ScheduledForecast.next_run <= now
    ).all()

def create_forecast_execution(
    db: Session,
    scheduled_forecast_id: int,
    execution_time: datetime,
    status: str,
    duration_seconds: Optional[int] = None,
    result_summary: Optional[str] = None,
    error_message: Optional[str] = None,
    forecast_data: Optional[str] = None
) -> ForecastExecution:
    """Create forecast execution record"""
    execution = ForecastExecution(
        scheduled_forecast_id=scheduled_forecast_id,
        execution_time=execution_time,
        status=status,
        duration_seconds=duration_seconds,
        result_summary=result_summary,
        error_message=error_message,
        forecast_data=forecast_data
    )
    db.add(execution)
    db.commit()
    db.refresh(execution)
    return execution

def get_executions_by_schedule_id(
    db: Session,
    schedule_id: int,
    limit: int = 100
) -> List[ForecastExecution]:
    """Get execution history for a scheduled forecast"""
    return db.query(ForecastExecution).filter(
        ForecastExecution.scheduled_forecast_id == schedule_id
    ).order_by(ForecastExecution.execution_time.desc()).limit(limit).all()

def get_latest_execution(
    db: Session,
    schedule_id: int
) -> Optional[ForecastExecution]:
    """Get latest execution for a scheduled forecast"""
    return db.query(ForecastExecution).filter(
        ForecastExecution.scheduled_forecast_id == schedule_id
    ).order_by(ForecastExecution.execution_time.desc()).first()
