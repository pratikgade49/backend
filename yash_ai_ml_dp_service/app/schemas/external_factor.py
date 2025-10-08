#!/usr/bin/env python3
"""
External factor schemas
"""

from pydantic import BaseModel
from typing import List, Dict, Any

class FredDataRequest(BaseModel):
    """Schema for FRED data request"""
    series_id: str
    start_date: str
    end_date: str

class FredDataResponse(BaseModel):
    """Schema for FRED data response"""
    series_id: str
    data: List[Dict[str, Any]]
    count: int
