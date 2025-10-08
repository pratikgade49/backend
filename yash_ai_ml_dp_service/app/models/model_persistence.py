#!/usr/bin/env python3
"""
Model persistence for trained forecasting models
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, LargeBinary, Float
from datetime import datetime
from app.core.database import Base

class SavedModel(Base):
    """Model for storing trained forecasting models"""
    __tablename__ = "saved_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_hash = Column(String(64), unique=True, nullable=False, index=True)
    algorithm = Column(String(100), nullable=False)
    config_hash = Column(String(64), nullable=False, index=True)
    model_data = Column(LargeBinary, nullable=False)
    model_metadata = Column(Text, nullable=True)
    accuracy = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, default=datetime.utcnow)
    use_count = Column(Integer, default=0)

class ModelAccuracyHistory(Base):
    """Model for tracking forecast accuracy over time"""
    __tablename__ = "model_accuracy_history"
    
    id = Column(Integer, primary_key=True, index=True)
    model_hash = Column(String(64), nullable=False, index=True)
    algorithm = Column(String(100), nullable=False)
    config_hash = Column(String(64), nullable=False)
    forecast_date = Column(DateTime, nullable=False)
    actual_values = Column(Text, nullable=True)
    predicted_values = Column(Text, nullable=True)
    accuracy = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
