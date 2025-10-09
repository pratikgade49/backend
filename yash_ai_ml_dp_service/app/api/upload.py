#!/usr/bin/env python3
"""
Upload endpoint for data files
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import pandas as pd
import io
from datetime import datetime

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.forecast import ForecastData
from app.services.dimension_service import DimensionService

router = APIRouter(tags=["Upload"])

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

        # Normalize column names to lowercase for case-insensitive matching
        df.columns = [col.lower() for col in df.columns]

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
            "inserted": records_added,
            "duplicates": duplicates_skipped,
            "totalRecords": len(df),
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
