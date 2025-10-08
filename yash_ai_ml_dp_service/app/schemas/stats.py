#!/usr/bin/env python3
"""
Statistics and data view schemas
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class DataViewRequest(BaseModel):
    """Schema for data view request"""
    product: Optional[str] = None
    customer: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = 100

class DataViewResponse(BaseModel):
    """Schema for data view response"""
    data: List[Dict[str, Any]]
    total_count: int
    filtered_count: int

class DatabaseStats(BaseModel):
    """Schema for database statistics"""
    total_records: int
    total_products: int
    total_customers: int
    total_locations: int
    date_range: Dict[str, str]
    total_users: int
    total_saved_forecasts: int
