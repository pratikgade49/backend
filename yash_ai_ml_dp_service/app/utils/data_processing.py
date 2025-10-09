#!/usr/bin/env python3
"""
Data processing utilities
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List

def parse_date_flexible(date_value) -> Optional[datetime]:
    """Parse date with multiple formats"""
    if pd.isna(date_value):
        return None
    
    if isinstance(date_value, datetime):
        return date_value
    
    # Try multiple date formats
    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_value), fmt)
        except ValueError:
            continue
    
    return None

def convert_to_numeric(value) -> Optional[float]:
    """Convert value to numeric, handling errors"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def detect_interval(dates: List[datetime]) -> str:
    """Detect the interval between dates"""
    if len(dates) < 2:
        return "day"
    
    sorted_dates = sorted(dates)
    diffs = [(sorted_dates[i+1] - sorted_dates[i]).days for i in range(len(sorted_dates)-1)]
    avg_diff = sum(diffs) / len(diffs)
    
    if avg_diff <= 1:
        return "day"
    elif avg_diff <= 7:
        return "week"
    elif avg_diff <= 31:
        return "month"
    elif avg_diff <= 92:
        return "quarter"
    else:
        return "year"

def fill_missing_dates(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Fill missing dates in a time series"""
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    
    # Create date range
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(date_range)
    
    # Forward fill for missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df.reset_index()

def process_uploaded_data(data, **kwargs):
    """Process uploaded data"""
    # Placeholder function
    return data
