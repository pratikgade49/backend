#!/usr/bin/env python3
"""
External factor schemas
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class FredDataRequest(BaseModel):
    """Schema for FRED data request"""
    series_ids: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class FredDataResponse(BaseModel):
    """Schema for FRED data response"""
    message: str
    inserted: int
    duplicates: int
    series_processed: int
    series_details: List[Dict[str, Any]]
