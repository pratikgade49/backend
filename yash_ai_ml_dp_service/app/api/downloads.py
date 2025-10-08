#!/usr/bin/env python3
"""
Excel download API routes
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import pandas as pd
import io
from datetime import datetime

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.repositories.dimension_repository import DimensionManager

router = APIRouter(tags=["Downloads"])

@router.post("/download_forecast_excel")
async def download_forecast_excel(
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Download single forecast data as Excel"""
    try:
        result = request['forecastResult']
        forecast_by = request.get('forecastBy', '')
        selected_item = request.get('selectedItem', '')

        hist = result['historicData']
        fore = result['forecastData']

        # Determine selected items
        product_name = ''
        customer_name = ''
        location_name = ''

        if result.get('combination'):
            if 'product_id' in result['combination'] and result['combination']['product_id'] is not None:
                product_name = DimensionManager.get_dimension_name(db, 'product', result['combination']['product_id'])
            elif 'product' in result['combination']:
                product_name = result['combination']['product']

            if 'customer_id' in result['combination'] and result['combination']['customer_id'] is not None:
                customer_name = DimensionManager.get_dimension_name(db, 'customer', result['combination']['customer_id'])
            elif 'customer' in result['combination']:
                customer_name = result['combination']['customer']

            if 'location_id' in result['combination'] and result['combination']['location_id'] is not None:
                location_name = DimensionManager.get_dimension_name(db, 'location', result['combination']['location_id'])
            elif 'location' in result['combination']:
                location_name = result['combination']['location']
        else:
            if forecast_by == 'product':
                product_name = selected_item
            elif forecast_by == 'customer':
                customer_name = selected_item
            elif forecast_by == 'location':
                location_name = selected_item

        # Create Excel data
        hist_rows = []
        fore_rows = []

        for item in hist:
            hist_rows.append({
                'Date': item['date'],
                'Quantity': item['quantity'],
                'Type': 'Historical',
                'Product': product_name,
                'Customer': customer_name,
                'Location': location_name
            })

        for item in fore:
            fore_rows.append({
                'Date': item['date'],
                'Quantity': item['quantity'],
                'Type': 'Forecast',
                'Product': product_name,
                'Customer': customer_name,
                'Location': location_name
            })

        df_hist = pd.DataFrame(hist_rows)
        df_fore = pd.DataFrame(fore_rows)
        df = pd.concat([df_hist, df_fore], ignore_index=True)

        # Create Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Forecast', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': ['Algorithm', 'Accuracy (%)', 'MAE', 'RMSE', 'Trend'],
                'Value': [
                    result.get('selectedAlgorithm', 'N/A'),
                    f"{result.get('accuracy', 0):.2f}",
                    f"{result.get('mae', 0):.2f}",
                    f"{result.get('rmse', 0):.2f}",
                    result.get('trend', 'N/A')
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)

        output.seek(0)
        
        filename = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Excel: {str(e)}")

@router.post("/download_multi_forecast_excel")
async def download_multi_forecast_excel(
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Download multi-forecast data as Excel"""
    try:
        results = request['multiForecastResult']['results']
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_rows = []
            for idx, result in enumerate(results, 1):
                combo = result.get('combination', {})
                
                product_name = combo.get('product', '')
                customer_name = combo.get('customer', '')
                location_name = combo.get('location', '')
                
                # Convert IDs to names if needed
                if 'product_id' in combo and combo['product_id']:
                    product_name = DimensionManager.get_dimension_name(db, 'product', combo['product_id'])
                if 'customer_id' in combo and combo['customer_id']:
                    customer_name = DimensionManager.get_dimension_name(db, 'customer', combo['customer_id'])
                if 'location_id' in combo and combo['location_id']:
                    location_name = DimensionManager.get_dimension_name(db, 'location', combo['location_id'])
                
                summary_rows.append({
                    'Combination': f"Combo {idx}",
                    'Product': product_name,
                    'Customer': customer_name,
                    'Location': location_name,
                    'Algorithm': result.get('selectedAlgorithm', 'N/A'),
                    'Accuracy (%)': f"{result.get('accuracy', 0):.2f}",
                    'Trend': result.get('trend', 'N/A')
                })
            
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual sheets for each combination
            for idx, result in enumerate(results, 1):
                hist_rows = []
                fore_rows = []
                
                combo = result.get('combination', {})
                product_name = combo.get('product', '')
                customer_name = combo.get('customer', '')
                location_name = combo.get('location', '')
                
                for item in result['historicData']:
                    hist_rows.append({
                        'Date': item['date'],
                        'Quantity': item['quantity'],
                        'Type': 'Historical'
                    })
                
                for item in result['forecastData']:
                    fore_rows.append({
                        'Date': item['date'],
                        'Quantity': item['quantity'],
                        'Type': 'Forecast'
                    })
                
                df_combo = pd.concat([pd.DataFrame(hist_rows), pd.DataFrame(fore_rows)], ignore_index=True)
                sheet_name = f"Combo_{idx}"[:31]  # Excel sheet name limit
                df_combo.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        
        filename = f"multi_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Excel: {str(e)}")
