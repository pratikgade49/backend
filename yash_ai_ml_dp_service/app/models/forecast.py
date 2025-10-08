#!/usr/bin/env python3
"""
Forecast data models
"""

from sqlalchemy import Column, Integer, String, Date, DateTime, DECIMAL, Text, UniqueConstraint
from datetime import datetime
from app.core.database import Base

class ForecastData(Base):
    """Model for storing forecast data from Excel uploads"""
    __tablename__ = "forecast_data"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Surrogate key references to dimension tables
    product_id = Column(Integer, nullable=True, index=True)
    customer_id = Column(Integer, nullable=True, index=True)
    location_id = Column(Integer, nullable=True, index=True)
    
    # Combination keys for faster filtering
    product_customer_id = Column(Integer, nullable=True, index=True)
    product_location_id = Column(Integer, nullable=True, index=True)
    customer_location_id = Column(Integer, nullable=True, index=True)
    product_customer_location_id = Column(Integer, nullable=True, index=True)
    
    # Core data fields
    quantity = Column(DECIMAL(15, 2), nullable=False)
    uom = Column(String(50), nullable=True)
    date = Column(Date, nullable=False)
    unit_price = Column(DECIMAL(15, 2), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Legacy string columns for backward compatibility
    product = Column(String(255), nullable=True)
    customer = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    
    __table_args__ = (
        UniqueConstraint(
            'product_id', 'customer_id', 'location_id', 'date',
            name='unique_forecast_record'
        ),
    )

class SavedForecastResult(Base):
    """Model for storing user's saved forecast results"""
    __tablename__ = "saved_forecast_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    forecast_config = Column(Text, nullable=False)
    forecast_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='unique_user_forecast_name'),
    )

class ForecastConfiguration(Base):
    """Model for storing forecast configurations"""
    __tablename__ = "forecast_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    forecast_by = Column(String(50), nullable=False)
    selected_item_id = Column(Integer, nullable=True)
    selected_item_name = Column(String(255), nullable=True)
    
    # For advanced mode
    selected_product_id = Column(Integer, nullable=True)
    selected_customer_id = Column(Integer, nullable=True)
    selected_location_id = Column(Integer, nullable=True)
    
    # Multi-select support
    selected_items_ids = Column(Text, nullable=True)
    algorithm = Column(String(100), nullable=False, default='best_fit')
    interval = Column(String(20), nullable=False, default='month')
    historic_period = Column(Integer, nullable=False, default=12)
    forecast_period = Column(Integer, nullable=False, default=6)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('name', name='unique_config_name'),
    )
