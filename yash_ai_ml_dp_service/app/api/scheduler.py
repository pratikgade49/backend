#!/usr/bin/env python3
"""
Scheduled forecast management API routes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.scheduler import (
    ScheduledForecastCreate,
    ScheduledForecastUpdate,
    ScheduledForecastResponse,
    ForecastExecutionResponse
)
from app.services.scheduler_service import SchedulerService
from app.models.scheduler import ScheduleFrequency, ScheduleStatus

router = APIRouter(prefix="/scheduled_forecasts", tags=["Scheduled Forecasts"])

@router.post("/", response_model=ScheduledForecastResponse)
async def create_scheduled_forecast(
    request: ScheduledForecastCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new scheduled forecast"""
    try:
        frequency = ScheduleFrequency(request.frequency)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid frequency")
    
    schedule = SchedulerService.create_schedule(
        db=db,
        user_id=current_user.id,
        name=request.name,
        config=request.forecast_config,
        frequency=frequency,
        start_date=request.start_date,
        end_date=request.end_date
    )
    
    return schedule

@router.get("/", response_model=List[ScheduledForecastResponse])
async def get_scheduled_forecasts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all scheduled forecasts for current user"""
    schedules = SchedulerService.get_user_schedules(db, current_user.id)
    return schedules

@router.get("/{forecast_id}", response_model=ScheduledForecastResponse)
async def get_scheduled_forecast(
    forecast_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific scheduled forecast"""
    schedule = SchedulerService.get_schedule(db, forecast_id, current_user.id)
    
    if not schedule:
        raise HTTPException(status_code=404, detail="Scheduled forecast not found")
    
    return schedule

@router.put("/{forecast_id}", response_model=ScheduledForecastResponse)
async def update_scheduled_forecast(
    forecast_id: int,
    request: ScheduledForecastUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a scheduled forecast"""
    update_data = request.dict(exclude_unset=True)
    
    schedule = SchedulerService.update_schedule(
        db=db,
        schedule_id=forecast_id,
        user_id=current_user.id,
        **update_data
    )
    
    if not schedule:
        raise HTTPException(status_code=404, detail="Scheduled forecast not found")
    
    return schedule

@router.delete("/{forecast_id}")
async def delete_scheduled_forecast(
    forecast_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a scheduled forecast"""
    success = SchedulerService.delete_schedule(db, forecast_id, current_user.id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Scheduled forecast not found")
    
    return {"message": "Scheduled forecast deleted successfully"}

@router.get("/{forecast_id}/executions", response_model=List[ForecastExecutionResponse])
async def get_forecast_executions(
    forecast_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get execution history for a scheduled forecast"""
    schedule = SchedulerService.get_schedule(db, forecast_id, current_user.id)
    
    if not schedule:
        raise HTTPException(status_code=404, detail="Scheduled forecast not found")
    
    executions = SchedulerService.get_executions(db, forecast_id)
    return executions

@router.get("/scheduler/status", prefix="", tags=["Scheduler"])
async def get_scheduler_status(current_user: User = Depends(get_current_user)):
    """Get scheduler status"""
    status = SchedulerService.get_status()
    return status
