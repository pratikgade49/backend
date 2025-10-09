#!/usr/bin/env python3
"""
External factors and FRED API integration routes
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Dict, Any, Optional
import pandas as pd
import io
from datetime import datetime
import requests

from app.core.database import get_db
from app.core.security import get_current_user, require_admin
from app.core.config import settings
from app.models.user import User
from app.models.external_factor import ExternalFactorData
from app.schemas.external_factor import FredDataRequest, FredDataResponse
from app.services.external_factor_service import ExternalFactorService
from app.utils.validation import DateRangeValidator


router = APIRouter(tags=["External Factors"])

@router.get("/external_factors")
async def get_external_factors(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get unique external factor names from database"""
    factors = ExternalFactorService.get_available_factors(db)
    return {"external_factors": factors}

@router.post("/fetch_fred_data", response_model=FredDataResponse)
async def fetch_fred_data(
    request: FredDataRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Fetch live economic data from FRED API and store in database"""
    if not settings.FRED_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="FRED API key not configured. Please set FRED_API_KEY environment variable."
        )

    # Clean the API key - remove any leading/trailing whitespace and special characters
    cleaned_api_key = settings.FRED_API_KEY.strip().lstrip('+')

    try:
        total_inserted = 0
        total_duplicates = 0
        series_details = []

        for series_id in request.series_ids:
            try:
                # Validate series_id (basic check)
                if not series_id or not isinstance(series_id, str):
                    series_details.append({
                        'series_id': series_id,
                        'status': 'error',
                        'message': 'Invalid series ID provided',
                        'inserted': 0
                    })
                    continue

                # Construct FRED API URL
                params = {
                    'series_id': series_id.upper().strip(),  # Ensure uppercase and clean
                    'api_key': cleaned_api_key,
                    'file_type': 'json',
                    'limit': 1000
                }

                # Add date parameters if provided
                if request.start_date:
                    # Ensure proper date format
                    if isinstance(request.start_date, str):
                        params['observation_start'] = request.start_date
                    else:
                        params['observation_start'] = request.start_date.strftime('%Y-%m-%d')

                if request.end_date:
                    # Ensure proper date format
                    if isinstance(request.end_date, str):
                        params['observation_end'] = request.end_date
                    else:
                        params['observation_end'] = request.end_date.strftime('%Y-%m-%d')

                # Make API request with better error handling
                print(f"Making FRED API request for series: {series_id}")
                print(f"Request URL: {settings.FRED_API_BASE_URL}")
                print(f"Request params: {params}")

                response = requests.get(
                    settings.FRED_API_BASE_URL, 
                    params=params, 
                    timeout=30,
                    headers={'User-Agent': 'YourApp/1.0'}  # Add user agent
                )

                # Log the actual URL being called (without API key for security)
                safe_params = {k: v if k != 'api_key' else '***HIDDEN***' for k, v in params.items()}
                print(f"Actual request URL: {response.url}")

                response.raise_for_status()

                data = response.json()

                # Check for API-level errors
                if 'error_message' in data:
                    series_details.append({
                        'series_id': series_id,
                        'status': 'error',
                        'message': f'FRED API error: {data["error_message"]}',
                        'inserted': 0
                    })
                    continue

                if 'observations' not in data:
                    series_details.append({
                        'series_id': series_id,
                        'status': 'error',
                        'message': 'No observations found in API response',
                        'inserted': 0
                    })
                    continue

                observations = data['observations']

                if not observations:
                    series_details.append({
                        'series_id': series_id,
                        'status': 'warning',
                        'message': 'No data available for the specified date range',
                        'inserted': 0
                    })
                    continue

                # Prepare records for insertion
                records_to_insert = []
                existing_records = set()

                # Get existing records for this series to avoid duplicates
                existing_query = db.query(ExternalFactorData.date, ExternalFactorData.factor_name).all()
                for rec in existing_query:
                    existing_records.add((rec.date, rec.factor_name))

                inserted_count = 0
                duplicate_count = 0
                skipped_count = 0

                for obs in observations:
                    try:
                        # Parse date and value with better error handling
                        obs_date = pd.to_datetime(obs['date']).date()

                        # Handle missing values (FRED uses '.' for missing data)
                        if obs['value'] == '.' or obs['value'] is None or obs['value'] == '':
                            skipped_count += 1
                            continue

                        obs_value = float(obs['value'])

                        if (obs_date, series_id) not in existing_records:
                            record_data = ExternalFactorData(
                                date=obs_date,
                                factor_name=series_id,
                                factor_value=obs_value
                            )
                            records_to_insert.append(record_data)
                            inserted_count += 1
                        else:
                            duplicate_count += 1

                    except (ValueError, TypeError) as e:
                        print(f"Error processing observation for {series_id}: {e}")
                        print(f"Problematic observation: {obs}")
                        skipped_count += 1
                        continue

                # Bulk insert new records
                if records_to_insert:
                    try:
                        db.bulk_save_objects(records_to_insert)
                        db.commit()
                        print(f"Successfully inserted {len(records_to_insert)} records for {series_id}")
                    except Exception as db_error:
                        db.rollback()
                        series_details.append({
                            'series_id': series_id,
                            'status': 'error',
                            'message': f'Database insertion failed: {str(db_error)}',
                            'inserted': 0
                        })
                        continue

                total_inserted += inserted_count
                total_duplicates += duplicate_count

                message_parts = [f'Successfully processed {len(observations)} observations']
                if skipped_count > 0:
                    message_parts.append(f'{skipped_count} skipped (missing values)')

                series_details.append({
                    'series_id': series_id,
                    'status': 'success',
                    'message': ', '.join(message_parts),
                    'inserted': inserted_count,
                    'duplicates': duplicate_count
                })

            except requests.RequestException as e:
                error_msg = f'API request failed: {str(e)}'
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        if 'error_message' in error_data:
                            error_msg += f' - {error_data["error_message"]}'
                    except:
                        error_msg += f' - HTTP {e.response.status_code}'

                series_details.append({
                    'series_id': series_id,
                    'status': 'error',
                    'message': error_msg,
                    'inserted': 0
                })
            except Exception as e:
                series_details.append({
                    'series_id': series_id,
                    'status': 'error',
                    'message': f'Processing failed: {str(e)}',
                    'inserted': 0
                })

        return FredDataResponse(
            message=f"FRED data fetch completed. Processed {len(request.series_ids)} series.",
            inserted=total_inserted,
            duplicates=total_duplicates,
            series_processed=len(request.series_ids),
            series_details=series_details
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching FRED data: {str(e)}")

@router.get("/fred_series_info")
async def get_fred_series_info(
    current_user: User = Depends(get_current_user)
):
    """Get information about popular FRED series for users"""
    popular_series = {
        "Economic Indicators": {
            "GDP": "Gross Domestic Product",
            "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
            "UNRATE": "Unemployment Rate",
            "FEDFUNDS": "Federal Funds Rate",
            "PAYEMS": "All Employees, Total Nonfarm",
            "INDPRO": "Industrial Production Index"
        },
        "Financial Markets": {
            "DGS10": "10-Year Treasury Constant Maturity Rate",
            "DGS3MO": "3-Month Treasury Constant Maturity Rate",
            "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
            "DEXJPUS": "Japan / U.S. Foreign Exchange Rate"
        },
        "Business & Trade": {
            "HOUST": "Housing Starts",
            "RSAFS": "Advance Retail Sales",
            "IMPGS": "Imports of Goods and Services",
            "EXPGS": "Exports of Goods and Services"
        }
    }

    return {
        "message": "Popular FRED series for economic forecasting",
        "series": popular_series,
        "note": "Visit https://fred.stlouisfed.org to explore more series"
    }

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


@router.post("/validate_factors")
def validate_external_factors(
    factor_names: List[str],
    db: Session = Depends(get_db)
):
    """Validate external factors against main data"""
    return DateRangeValidator.validate_external_factors(db, factor_names)

@router.post("/fetch_fred_data")
async def fetch_fred_data_root(
    request: FredDataRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Root-level fetch FRED data endpoint"""
    return await fetch_fred_data(request, db, current_user)

@router.get("/fred_series_info")
async def get_fred_series_info_root():
    """Root-level FRED series info endpoint"""
    return await get_fred_series_info()

