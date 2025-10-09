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
    page: int = 1
    page_size: int = 50

class DataViewResponse(BaseModel):
    """Schema for data view response"""
    data: List[Dict[str, Any]]
    total_records: int
    page: int
    page_size: int
    total_pages: int

class DatabaseStats(BaseModel):
    """Schema for database statistics"""
    totalRecords: int
    uniqueProducts: int
    uniqueCustomers: int
    uniqueLocations: int
    dateRange: Dict[str, str]
    totalUsers: int
    totalSavedForecasts: int
