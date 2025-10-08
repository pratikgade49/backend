#!/usr/bin/env python3
"""
Forecast data repository
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from datetime import datetime, date
from app.models.forecast import ForecastData, SavedForecastResult, ForecastConfiguration
import json

def save_forecast_data_batch(db: Session, forecast_records: List[ForecastData]) -> int:
    """Batch save forecast data records"""
    db.bulk_save_objects(forecast_records)
    db.commit()
    return len(forecast_records)

def get_forecast_data_by_filters(
    db: Session,
    product_id: Optional[int] = None,
    customer_id: Optional[int] = None,
    location_id: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    product_ids: Optional[List[int]] = None,
    customer_ids: Optional[List[int]] = None,
    location_ids: Optional[List[int]] = None
) -> List[ForecastData]:
    """Get forecast data with filters"""
    query = db.query(ForecastData)
    
    # Single item filters
    if product_id:
        query = query.filter(ForecastData.product_id == product_id)
    if customer_id:
        query = query.filter(ForecastData.customer_id == customer_id)
    if location_id:
        query = query.filter(ForecastData.location_id == location_id)
    
    # Multi-item filters
    if product_ids:
        query = query.filter(ForecastData.product_id.in_(product_ids))
    if customer_ids:
        query = query.filter(ForecastData.customer_id.in_(customer_ids))
    if location_ids:
        query = query.filter(ForecastData.location_id.in_(location_ids))
    
    # Date range filters
    if start_date:
        query = query.filter(ForecastData.date >= start_date)
    if end_date:
        query = query.filter(ForecastData.date <= end_date)
    
    return query.order_by(ForecastData.date).all()

def get_forecast_data_aggregated(
    db: Session,
    product_id: Optional[int] = None,
    customer_id: Optional[int] = None,
    location_id: Optional[int] = None,
    interval: str = 'month'
) -> List[Dict[str, Any]]:
    """Get aggregated forecast data"""
    query = db.query(
        ForecastData.date,
        func.sum(ForecastData.quantity).label('total_quantity')
    )
    
    if product_id:
        query = query.filter(ForecastData.product_id == product_id)
    if customer_id:
        query = query.filter(ForecastData.customer_id == customer_id)
    if location_id:
        query = query.filter(ForecastData.location_id == location_id)
    
    query = query.group_by(ForecastData.date).order_by(ForecastData.date)
    
    return [{"date": row.date, "quantity": float(row.total_quantity)} for row in query.all()]

def get_forecast_data_date_range(db: Session) -> Dict[str, Optional[date]]:
    """Get min and max dates from forecast data"""
    result = db.query(
        func.min(ForecastData.date).label('min_date'),
        func.max(ForecastData.date).label('max_date')
    ).first()
    
    return {
        "min_date": result.min_date,
        "max_date": result.max_date
    }

def get_forecast_data_count(db: Session) -> int:
    """Get total forecast data count"""
    return db.query(ForecastData).count()

def delete_forecast_data_by_filters(
    db: Session,
    product_id: Optional[int] = None,
    customer_id: Optional[int] = None,
    location_id: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> int:
    """Delete forecast data by filters"""
    query = db.query(ForecastData)
    
    if product_id:
        query = query.filter(ForecastData.product_id == product_id)
    if customer_id:
        query = query.filter(ForecastData.customer_id == customer_id)
    if location_id:
        query = query.filter(ForecastData.location_id == location_id)
    if start_date:
        query = query.filter(ForecastData.date >= start_date)
    if end_date:
        query = query.filter(ForecastData.date <= end_date)
    
    count = query.count()
    query.delete(synchronize_session=False)
    db.commit()
    return count

def save_forecast_result(
    db: Session,
    user_id: int,
    name: str,
    description: Optional[str],
    forecast_config: dict,
    forecast_data: dict
) -> SavedForecastResult:
    """Save forecast result for user"""
    saved_forecast = SavedForecastResult(
        user_id=user_id,
        name=name,
        description=description,
        forecast_config=json.dumps(forecast_config),
        forecast_data=json.dumps(forecast_data)
    )
    db.add(saved_forecast)
    db.commit()
    db.refresh(saved_forecast)
    return saved_forecast

def get_saved_forecasts_by_user(db: Session, user_id: int) -> List[SavedForecastResult]:
    """Get all saved forecasts for a user"""
    return db.query(SavedForecastResult).filter(
        SavedForecastResult.user_id == user_id
    ).order_by(SavedForecastResult.created_at.desc()).all()

def get_saved_forecast_by_id(db: Session, forecast_id: int, user_id: int) -> Optional[SavedForecastResult]:
    """Get saved forecast by ID for specific user"""
    return db.query(SavedForecastResult).filter(
        SavedForecastResult.id == forecast_id,
        SavedForecastResult.user_id == user_id
    ).first()

def delete_saved_forecast(db: Session, forecast_id: int, user_id: int) -> bool:
    """Delete saved forecast"""
    forecast = db.query(SavedForecastResult).filter(
        SavedForecastResult.id == forecast_id,
        SavedForecastResult.user_id == user_id
    ).first()
    
    if forecast:
        db.delete(forecast)
        db.commit()
        return True
    return False

def get_saved_forecast_count(db: Session, user_id: Optional[int] = None) -> int:
    """Get count of saved forecasts"""
    query = db.query(SavedForecastResult)
    if user_id:
        query = query.filter(SavedForecastResult.user_id == user_id)
    return query.count()

def save_forecast_configuration(
    db: Session,
    name: str,
    description: Optional[str],
    config: dict
) -> ForecastConfiguration:
    """Save forecast configuration"""
    forecast_config = ForecastConfiguration(
        name=name,
        description=description,
        forecast_by=config.get('forecastBy'),
        selected_item_id=config.get('selectedItemId'),
        selected_item_name=config.get('selectedItem'),
        selected_product_id=config.get('selectedProductId'),
        selected_customer_id=config.get('selectedCustomerId'),
        selected_location_id=config.get('selectedLocationId'),
        selected_items_ids=json.dumps(config.get('selectedItemsIds', [])),
        algorithm=config.get('algorithm', 'best_fit'),
        interval=config.get('interval', 'month'),
        historic_period=config.get('historicPeriod', 12),
        forecast_period=config.get('forecastPeriod', 6)
    )
    db.add(forecast_config)
    db.commit()
    db.refresh(forecast_config)
    return forecast_config

def get_forecast_configuration_by_name(db: Session, name: str) -> Optional[ForecastConfiguration]:
    """Get forecast configuration by name"""
    return db.query(ForecastConfiguration).filter(
        ForecastConfiguration.name == name
    ).first()

def get_all_forecast_configurations(db: Session) -> List[ForecastConfiguration]:
    """Get all forecast configurations"""
    return db.query(ForecastConfiguration).order_by(
        ForecastConfiguration.created_at.desc()
    ).all()

def delete_forecast_configuration(db: Session, config_id: int) -> bool:
    """Delete forecast configuration"""
    config = db.query(ForecastConfiguration).filter(
        ForecastConfiguration.id == config_id
    ).first()
    
    if config:
        db.delete(config)
        db.commit()
        return True
    return False
