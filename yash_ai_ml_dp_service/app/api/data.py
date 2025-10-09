#!/usr/bin/env python3
"""
Data upload and management API routes
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import pandas as pd
import io
from datetime import datetime

from app.core.database import get_db
from app.core.security import get_current_user, require_admin
from app.models.user import User
from app.models.forecast import ForecastData
from app.services.dimension_service import DimensionService
from app.utils.data_processing import process_uploaded_data

router = APIRouter(tags=["Data Management"])

@router.post("/upload")
async def upload_data(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload forecast data from CSV or Excel file"""
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
        
        required_columns = ['date', 'quantity']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns. Need: {required_columns}"
            )
        
        df['date'] = pd.to_datetime(df['date'])
        
        records_added = 0
        duplicates_skipped = 0
        
        for _, row in df.iterrows():
            product_id = None
            customer_id = None
            location_id = None
            
            if 'product' in df.columns and pd.notna(row['product']):
                product_id = DimensionService.get_or_create_product(db, str(row['product']))
            
            if 'customer' in df.columns and pd.notna(row['customer']):
                customer_id = DimensionService.get_or_create_customer(db, str(row['customer']))
            
            if 'location' in df.columns and pd.notna(row['location']):
                location_id = DimensionService.get_or_create_location(db, str(row['location']))
            
            try:
                forecast_data = ForecastData(
                    date=row['date'].date(),
                    quantity=float(row['quantity']),
                    product_id=product_id,
                    customer_id=customer_id,
                    location_id=location_id
                )
                db.add(forecast_data)
                db.commit()
                records_added += 1
            except IntegrityError:
                db.rollback()
                duplicates_skipped += 1
                continue
        
        return {
            "message": "Data uploaded successfully",
            "records_added": records_added,
            "duplicates_skipped": duplicates_skipped,
            "total_rows": len(df)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@router.get("/products")
async def get_products(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all available products"""
    products = DimensionService.get_all_products(db)
    return {"products": products}

@router.get("/customers")
async def get_customers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all available customers"""
    customers = DimensionService.get_all_customers(db)
    return {"customers": customers}

@router.get("/locations")
async def get_locations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all available locations"""
    locations = DimensionService.get_all_locations(db)
    return {"locations": locations}

@router.get("/dimensions")
async def get_all_dimensions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all available dimensions in one call"""
    return {
        "products": DimensionService.get_all_products(db),
        "customers": DimensionService.get_all_customers(db),
        "locations": DimensionService.get_all_locations(db)
    }