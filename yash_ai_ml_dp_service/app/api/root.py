#!/usr/bin/env python3
"""
Root-level API routes for convenience endpoints
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter(tags=["Root"])

@router.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "message": "Multi-variant Forecasting API is running",
        "status": "healthy",
        "version": "3.0.0"
    }

@router.get("/algorithms")
async def get_algorithms():
    """Get available forecasting algorithms"""
    return {
        "algorithms": {
            "ARIMA": "Auto Regressive Integrated Moving Average",
            "SARIMA": "Seasonal ARIMA",
            "ETS": "Exponential Smoothing",
            "Prophet": "Facebook Prophet",
            "LSTM": "Long Short-Term Memory Neural Network",
            "XGBoost": "Extreme Gradient Boosting"
        }
    }

@router.post("/best_fit_recommendation")
async def best_fit_recommendation_root(
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Root-level best fit recommendation (delegates to forecast router)"""
    from app.api.forecast import get_best_fit_recommendation
    from app.schemas.forecast import BestFitRecommendationRequest, ForecastConfig
    
    config = ForecastConfig(**payload['config'])
    request = BestFitRecommendationRequest(config=config)
    
    return await get_best_fit_recommendation(request, db, current_user)
