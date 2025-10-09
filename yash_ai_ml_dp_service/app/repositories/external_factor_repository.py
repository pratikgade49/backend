
#!/usr/bin/env python3
"""
External factor data repository
"""

from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date
from app.models.external_factor import ExternalFactorData

def save_external_factor_data(
    db: Session,
    factor_name: str,
    data_points: List[Dict]
) -> int:
    """Save external factor data points"""
    records = []
    for point in data_points:
        record = ExternalFactorData(
            date=point['date'],
            factor_name=factor_name,
            factor_value=point['value']
        )
        records.append(record)
    
    db.bulk_save_objects(records)
    db.commit()
    return len(records)

def get_external_factor_data(
    db: Session,
    factor_name: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> List[ExternalFactorData]:
    """Get external factor data by name and date range"""
    query = db.query(ExternalFactorData).filter(
        ExternalFactorData.factor_name == factor_name
    )
    
    if start_date:
        query = query.filter(ExternalFactorData.date >= start_date)
    if end_date:
        query = query.filter(ExternalFactorData.date <= end_date)
    
    return query.order_by(ExternalFactorData.date).all()

def get_distinct_factor_names(db: Session) -> List[str]:
    """Get list of distinct factor names"""
    result = db.query(ExternalFactorData.factor_name).distinct().all()
    return [row[0] for row in result]

def get_external_factor_date_range(
    db: Session,
    factor_name: Optional[str] = None
) -> Dict[str, Optional[date]]:
    """Get date range for external factors"""
    query = db.query(
        func.min(ExternalFactorData.date).label('min_date'),
        func.max(ExternalFactorData.date).label('max_date')
    )
    
    if factor_name:
        query = query.filter(ExternalFactorData.factor_name == factor_name)
    
    result = query.first()
    return {
        "min_date": result.min_date,
        "max_date": result.max_date
    }

def delete_external_factor_by_name(db: Session, factor_name: str) -> int:
    """Delete all data for a specific external factor"""
    count = db.query(ExternalFactorData).filter(
        ExternalFactorData.factor_name == factor_name
    ).count()
    
    db.query(ExternalFactorData).filter(
        ExternalFactorData.factor_name == factor_name
    ).delete()
    db.commit()
    
    return count

def get_external_factor_count(db: Session) -> int:
    """Get total external factor data count"""
    return db.query(ExternalFactorData).count()

def get_all_external_factors(db: Session) -> List[ExternalFactorData]:
    """Get all external factor data"""
    return db.query(ExternalFactorData).order_by(
        ExternalFactorData.factor_name, ExternalFactorData.date
    ).all()

def get_external_factor_by_name(db: Session, factor_name: str) -> List[ExternalFactorData]:
    """Get external factor data by name"""
    return db.query(ExternalFactorData).filter(
        ExternalFactorData.factor_name == factor_name
    ).order_by(ExternalFactorData.date).all()

def create_external_factor(
    db: Session,
    factor_name: str,
    date: date,
    factor_value: float
) -> ExternalFactorData:
    """Create a new external factor record"""
    factor = ExternalFactorData(
        factor_name=factor_name,
        date=date,
        factor_value=factor_value
    )
    db.add(factor)
    db.commit()
    db.refresh(factor)
    return factor

def get_external_factors_for_date_range(
    db: Session,
    start_date: date,
    end_date: date,
    factor_name: Optional[str] = None
) -> List[ExternalFactorData]:
    """Get external factors within a date range"""
    query = db.query(ExternalFactorData).filter(
        ExternalFactorData.date >= start_date,
        ExternalFactorData.date <= end_date
    )
    
    if factor_name:
        query = query.filter(ExternalFactorData.factor_name == factor_name)
    
    return query.order_by(ExternalFactorData.date).all()
