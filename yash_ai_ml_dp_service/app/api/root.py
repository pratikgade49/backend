#!/usr/bin/env python3
"""
Root endpoints - health check and algorithm list
"""

from fastapi import APIRouter
from app.services.forecasting_service import ForecastingEngine

router = APIRouter(tags=["Root"])

@router.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "message": "Advanced Multi-variant Forecasting API with PostgreSQL is running",
        "algorithms": list(ForecastingEngine.ALGORITHMS.values())
    }

@router.get("/algorithms")
async def get_algorithms():
    """Get list of available forecasting algorithms"""
    return {
        "algorithms": ForecastingEngine.ALGORITHMS,
        "count": len(ForecastingEngine.ALGORITHMS)
    }
