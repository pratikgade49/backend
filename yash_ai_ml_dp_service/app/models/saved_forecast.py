#!/usr/bin/env python3
"""
Saved forecast result model
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.core.database import Base

class SavedForecastResult(Base):
    """Saved forecast results"""
    __tablename__ = "saved_forecast_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    forecast_config = Column(Text, nullable=False)  # JSON string
    forecast_data = Column(Text, nullable=False)    # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
