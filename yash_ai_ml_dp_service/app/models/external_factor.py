#!/usr/bin/env python3
"""
External factor data model
"""

from sqlalchemy import Column, Integer, String, Float, Date, DateTime, UniqueConstraint
from datetime import datetime
from app.core.database import Base

class ExternalFactorData(Base):
    """Model for storing external factor data"""
    __tablename__ = 'external_factor_data'
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    factor_name = Column(String(255), index=True)
    factor_value = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('date', 'factor_name', name='unique_external_factor_record'),
    )
