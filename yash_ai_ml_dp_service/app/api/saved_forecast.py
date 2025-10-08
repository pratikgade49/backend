#!/usr/bin/env python3
"""
Saved forecast management API routes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import json

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.saved_forecast import SavedForecastResult
from app.repositories.dimension_repository import DimensionManager

router = APIRouter(prefix="/saved_forecasts", tags=["Saved Forecasts"])

@router.get("/")
async def get_saved_forecasts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all saved forecasts for the current user"""
    try:
        saved_forecasts = db.query(SavedForecastResult).filter(
            SavedForecastResult.user_id == current_user.id
        ).order_by(SavedForecastResult.created_at.desc()).all()

        result = []
        for forecast in saved_forecasts:
            try:
                forecast_config_dict = json.loads(forecast.forecast_config)
                forecast_data_dict = json.loads(forecast.forecast_data)

                # Convert IDs back to names for display
                if 'selectedItem' in forecast_config_dict and forecast_config_dict['selectedItem'] is None and 'selected_item_id' in forecast_config_dict:
                    forecast_config_dict['selectedItem'] = DimensionManager.get_dimension_name(db, forecast_config_dict['forecastBy'], forecast_config_dict['selected_item_id'])
                if 'selectedProduct' in forecast_config_dict and forecast_config_dict['selectedProduct'] is None and 'selected_product_id' in forecast_config_dict:
                    forecast_config_dict['selectedProduct'] = DimensionManager.get_dimension_name(db, 'product', forecast_config_dict['selected_product_id'])
                if 'selectedCustomer' in forecast_config_dict and forecast_config_dict['selectedCustomer'] is None and 'selected_customer_id' in forecast_config_dict:
                    forecast_config_dict['selectedCustomer'] = DimensionManager.get_dimension_name(db, 'customer', forecast_config_dict['selected_customer_id'])
                if 'selectedLocation' in forecast_config_dict and forecast_config_dict['selectedLocation'] is None and 'selected_location_id' in forecast_config_dict:
                    forecast_config_dict['selectedLocation'] = DimensionManager.get_dimension_name(db, 'location', forecast_config_dict['selected_location_id'])
                
                # Handle multi-select items
                if 'selectedItems' in forecast_config_dict and forecast_config_dict['selectedItems'] is None and 'selected_items_ids' in forecast_config_dict and forecast_config_dict['selected_items_ids']:
                    selected_ids = json.loads(forecast_config_dict['selected_items_ids'])
                    selected_names = []
                    for item_id in selected_ids:
                        selected_names.append(DimensionManager.get_dimension_name(db, forecast_config_dict['forecastBy'], item_id))
                    forecast_config_dict['selectedItems'] = selected_names

                # Handle combination names in forecast_data
                if 'results' in forecast_data_dict and forecast_data_dict['results']:
                    for res in forecast_data_dict['results']:
                        if 'combination' in res and res['combination']:
                            if 'product_id' in res['combination'] and res['combination']['product_id'] is not None:
                                res['combination']['product'] = DimensionManager.get_dimension_name(db, 'product', res['combination']['product_id'])
                            if 'customer_id' in res['combination'] and res['combination']['customer_id'] is not None:
                                res['combination']['customer'] = DimensionManager.get_dimension_name(db, 'customer', res['combination']['customer_id'])
                            if 'location_id' in res['combination'] and res['combination']['location_id'] is not None:
                                res['combination']['location'] = DimensionManager.get_dimension_name(db, 'location', res['combination']['location_id'])

                result.append({
                    'id': forecast.id,
                    'user_id': forecast.user_id,
                    'name': forecast.name,
                    'description': forecast.description,
                    'forecast_config': forecast_config_dict,
                    'forecast_data': forecast_data_dict,
                    'created_at': forecast.created_at.isoformat(),
                    'updated_at': forecast.updated_at.isoformat()
                })
            except Exception as e:
                print(f"Error parsing saved forecast {forecast.id}: {e}")
                continue

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting saved forecasts: {str(e)}")

@router.post("/")
async def save_forecast(
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Save a forecast result"""
    try:
        # Convert names to IDs in forecast_config before saving
        config_dict = request['forecast_config']
        if config_dict.get('selectedItem'):
            config_dict['selected_item_id'] = DimensionManager.get_dimension_id(db, config_dict['forecastBy'], config_dict['selectedItem'])
        if config_dict.get('selectedProduct'):
            config_dict['selected_product_id'] = DimensionManager.get_dimension_id(db, 'product', config_dict['selectedProduct'])
        if config_dict.get('selectedCustomer'):
            config_dict['selected_customer_id'] = DimensionManager.get_dimension_id(db, 'customer', config_dict['selectedCustomer'])
        if config_dict.get('selectedLocation'):
            config_dict['selected_location_id'] = DimensionManager.get_dimension_id(db, 'location', config_dict['selectedLocation'])

        saved_forecast = SavedForecastResult(
            user_id=current_user.id,
            name=request['name'],
            description=request.get('description', ''),
            forecast_config=json.dumps(config_dict),
            forecast_data=json.dumps(request['forecast_data'])
        )

        db.add(saved_forecast)
        db.commit()
        db.refresh(saved_forecast)

        return {
            'id': saved_forecast.id,
            'message': 'Forecast saved successfully'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving forecast: {str(e)}")

@router.delete("/{forecast_id}")
async def delete_saved_forecast(
    forecast_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a saved forecast"""
    try:
        forecast = db.query(SavedForecastResult).filter(
            SavedForecastResult.id == forecast_id,
            SavedForecastResult.user_id == current_user.id
        ).first()

        if not forecast:
            raise HTTPException(status_code=404, detail="Saved forecast not found")

        db.delete(forecast)
        db.commit()

        return {"message": "Saved forecast deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting saved forecast: {str(e)}")
