#!/usr/bin/env python3
"""
Scheduler service - handles scheduled forecast management
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from app.repositories.scheduler_repository import (
    create_scheduled_forecast,
    get_scheduled_forecast_by_id,
    get_user_scheduled_forecasts,
    update_scheduled_forecast,
    delete_scheduled_forecast,
    create_forecast_execution,
    get_forecast_executions,
    update_execution_status
)
from app.models.scheduler import ScheduledForecast, ForecastExecution, ScheduleFrequency, ScheduleStatus

class SchedulerService:
    """Service for managing scheduled forecasts"""

    @staticmethod
    def create_schedule(db: Session, user_id: int, name: str, config: dict,
                       frequency: ScheduleFrequency, start_date: datetime,
                       end_date: Optional[datetime] = None) -> ScheduledForecast:
        """Create a new scheduled forecast"""
        return create_scheduled_forecast(
            db=db,
            user_id=user_id,
            name=name,
            config=config,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date
        )

    @staticmethod
    def get_schedule(db: Session, schedule_id: int, user_id: int) -> Optional[ScheduledForecast]:
        """Get a scheduled forecast by ID"""
        schedule = get_scheduled_forecast_by_id(db, schedule_id)
        if schedule and schedule.user_id == user_id:
            return schedule
        return None

    @staticmethod
    def get_user_schedules(db: Session, user_id: int) -> List[ScheduledForecast]:
        """Get all schedules for a user"""
        return get_user_scheduled_forecasts(db, user_id)

    @staticmethod
    def update_schedule(db: Session, schedule_id: int, user_id: int,
                       name: Optional[str] = None,
                       config: Optional[dict] = None,
                       frequency: Optional[ScheduleFrequency] = None,
                       status: Optional[ScheduleStatus] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> Optional[ScheduledForecast]:
        """Update a scheduled forecast"""
        schedule = SchedulerService.get_schedule(db, schedule_id, user_id)
        if not schedule:
            return None

        return update_scheduled_forecast(
            db=db,
            schedule_id=schedule_id,
            name=name,
            config=config,
            frequency=frequency,
            status=status,
            start_date=start_date,
            end_date=end_date
        )

    @staticmethod
    def delete_schedule(db: Session, schedule_id: int, user_id: int) -> bool:
        """Delete a scheduled forecast"""
        schedule = SchedulerService.get_schedule(db, schedule_id, user_id)
        if not schedule:
            return False

        return delete_scheduled_forecast(db, schedule_id)

    @staticmethod
    def record_execution(db: Session, schedule_id: int, status: str,
                        result_data: Optional[dict] = None,
                        error_message: Optional[str] = None) -> ForecastExecution:
        """Record a forecast execution"""
        return create_forecast_execution(
            db=db,
            schedule_id=schedule_id,
            status=status,
            result_data=result_data,
            error_message=error_message
        )

    @staticmethod
    def get_schedule_executions(db: Session, schedule_id: int, user_id: int,
                               limit: int = 50) -> List[ForecastExecution]:
        """Get execution history for a schedule"""
        schedule = SchedulerService.get_schedule(db, schedule_id, user_id)
        if not schedule:
            return []

        return get_forecast_executions(db, schedule_id, limit)

    @staticmethod
    def update_execution(db: Session, execution_id: int, status: str,
                        result_data: Optional[dict] = None,
                        error_message: Optional[str] = None) -> Optional[ForecastExecution]:
        """Update an execution record"""
        return update_execution_status(
            db=db,
            execution_id=execution_id,
            status=status,
            result_data=result_data,
            error_message=error_message
        )
