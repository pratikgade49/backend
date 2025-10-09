#!/usr/bin/env python3
"""
Background scheduler thread for automated forecast execution
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.scheduler import ScheduledForecast, ForecastExecution, ScheduleFrequency, ScheduleStatus
from app.services.forecasting_service import ForecastingEngine
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastScheduler:
    """Background scheduler for automated forecast execution"""
    
    def __init__(self):
        self.running = False
        self.scheduler_thread = None
        self.check_interval = 60  # Check every minute
        
    def start(self):
        """Start the scheduler thread"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
            
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info("Forecast scheduler started")
        
    def stop(self):
        """Stop the scheduler thread"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Forecast scheduler stopped")
        
    def _run_scheduler(self):
        """Main scheduler loop - runs in background thread"""
        while self.running:
            try:
                self._check_and_execute_forecasts()
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
            
            # Sleep for check_interval seconds
            time.sleep(self.check_interval)
    
    def _check_and_execute_forecasts(self):
        """Check for due forecasts and execute them"""
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            
            # Find all active schedules that are due
            due_schedules = db.query(ScheduledForecast).filter(
                ScheduledForecast.status == 'active',
                ScheduledForecast.next_run <= now
            ).all()
            
            for schedule in due_schedules:
                logger.info(f"Executing scheduled forecast: {schedule.name} (ID: {schedule.id})")
                self._execute_forecast(db, schedule)
                
        except Exception as e:
            logger.error(f"Error checking forecasts: {e}")
        finally:
            db.close()
    
    def _execute_forecast(self, db: Session, schedule: ScheduledForecast):
        """Execute a single scheduled forecast"""
        execution_start = datetime.utcnow()
        
        try:
            # Parse forecast config
            config = json.loads(schedule.forecast_config)
            
            # Create execution record
            execution = ForecastExecution(
                scheduled_forecast_id=schedule.id,
                execution_time=execution_start,
                status='running'
            )
            db.add(execution)
            db.commit()
            
            # Execute the forecast
            forecasting_service = ForecastingEngine()
            result = forecasting_service.generate_forecast(db, config)
            
            # Calculate duration
            execution_end = datetime.utcnow()
            duration = int((execution_end - execution_start).total_seconds())
            
            # Update execution record with success
            execution.status = 'success'
            execution.duration_seconds = duration
            execution.result_summary = json.dumps({
                'forecast_points': len(result.get('forecast', [])),
                'algorithm_used': result.get('best_algorithm', config.get('algorithm')),
                'accuracy': result.get('accuracy')
            })
            execution.forecast_data = json.dumps(result)
            
            # Update schedule
            schedule.last_run = execution_start
            schedule.next_run = self._calculate_next_run(schedule.next_run, schedule.frequency)
            schedule.run_count += 1
            schedule.success_count += 1
            schedule.last_error = None
            
            # Check if schedule should be completed
            if schedule.end_date and schedule.next_run > schedule.end_date:
                schedule.status = 'completed'
            
            db.commit()
            logger.info(f"Successfully executed forecast {schedule.name}")
            
        except Exception as e:
            logger.error(f"Error executing forecast {schedule.name}: {e}")
            
            # Update execution record with failure
            execution_end = datetime.utcnow()
            duration = int((execution_end - execution_start).total_seconds())
            
            execution.status = 'failed'
            execution.duration_seconds = duration
            execution.error_message = str(e)
            
            # Update schedule
            schedule.last_run = execution_start
            schedule.next_run = self._calculate_next_run(schedule.next_run, schedule.frequency)
            schedule.run_count += 1
            schedule.failure_count += 1
            schedule.last_error = str(e)
            schedule.status = 'failed'
            
            db.commit()
    
    def _calculate_next_run(self, last_run: datetime, frequency: str) -> datetime:
        """Calculate next run time based on frequency"""
        if frequency == 'daily' or frequency == ScheduleFrequency.DAILY.value:
            return last_run + timedelta(days=1)
        elif frequency == 'weekly' or frequency == ScheduleFrequency.WEEKLY.value:
            return last_run + timedelta(weeks=1)
        elif frequency == 'monthly' or frequency == ScheduleFrequency.MONTHLY.value:
            # Add approximately one month
            if last_run.month == 12:
                return last_run.replace(year=last_run.year + 1, month=1)
            else:
                return last_run.replace(month=last_run.month + 1)
        else:
            # Default to daily
            return last_run + timedelta(days=1)

# Global scheduler instance
_scheduler = ForecastScheduler()

def start_scheduler():
    """Start the global scheduler"""
    _scheduler.start()

def stop_scheduler():
    """Stop the global scheduler"""
    _scheduler.stop()

def get_scheduler_status():
    """Get scheduler status"""
    return {
        "running": _scheduler.running,
        "check_interval": _scheduler.check_interval
    }
