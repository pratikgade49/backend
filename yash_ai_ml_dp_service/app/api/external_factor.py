#!/usr/bin/env python3
"""
External factors and FRED API integration routes
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Dict, Any
import pandas as pd
import io
from datetime import datetime

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User
from app.models.external_factor import ExternalFactorData
from app.schemas.external_factor import FredDataRequest, FredDataResponse
from app.services.external_factor_service import ExternalFactorService

router = APIRouter(tags=["External Factors"])

@router.get("/external_factors")
async def get_external_factors(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get unique external factor names from database"""
    factors = ExternalFactorService.get_unique_factors(db)
    return {"external_factors": factors}

@router.post("/fetch_fred_data", response_model=FredDataResponse)
async def fetch_fred_data(
    request: FredDataRequest,
    current_user: User = Depends(get_current_user)
):
    """Fetch data from FRED API"""
    try:
        data = ExternalFactorService.fetch_fred_data(
            series_id=request.series_id,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        return FredDataResponse(
            series_id=request.series_id,
            data=data,
            count=len(data)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/fred_series_info")
async def get_fred_series_info(
    series_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get information about a FRED series"""
    try:
        info = ExternalFactorService.get_fred_series_info(series_id)
        return info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/upload_external_factors")
async def upload_external_factors(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload external factors from CSV or Excel file"""
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload CSV or Excel file."
        )
    
    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        required_columns = ['date', 'factor_name', 'value']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns. Need: {required_columns}"
            )
        
        df['date'] = pd.to_datetime(df['date'])
        
        records_added = 0
        duplicates_skipped = 0
        
        for _, row in df.iterrows():
            try:
                factor = ExternalFactorData(
                    date=row['date'].date(),
                    factor_name=str(row['factor_name']),
                    value=float(row['value'])
                )
                db.add(factor)
                db.commit()
                records_added += 1
            except IntegrityError:
                db.rollback()
                duplicates_skipped += 1
                continue
        
        return {
            "message": "External factors uploaded successfully",
            "records_added": records_added,
            "duplicates_skipped": duplicates_skipped,
            "total_rows": len(df)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
