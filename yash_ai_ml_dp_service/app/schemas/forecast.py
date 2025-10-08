#!/usr/bin/env python3
"""
Forecast schemas for request/response validation
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ForecastConfig(BaseModel):
    """Schema for forecast configuration"""
    forecastBy: str
    selectedItem: Optional[str] = None
    selectedProduct: Optional[str] = None
    selectedCustomer: Optional[str] = None
    selectedLocation: Optional[str] = None
    selectedItems: Optional[List[str]] = None
    selectedProducts: Optional[List[str]] = None
    selectedCustomers: Optional[List[str]] = None
    selectedLocations: Optional[List[str]] = None
    algorithm: str = "best_fit"
    interval: str = "month"
    historicPeriod: int = 12
    forecastPeriod: int = 6
    multiSelect: Optional[bool] = False
    externalFactors: Optional[List[str]] = []

class DataPoint(BaseModel):
    """Schema for a single data point"""
    date: str
    value: float

class AlgorithmResult(BaseModel):
    """Schema for algorithm result"""
    algorithm: str
    accuracy: float
    mae: float
    rmse: float
    forecast: List[DataPoint]

class ForecastResult(BaseModel):
    """Schema for single forecast result"""
    historicalData: List[DataPoint]
    forecastData: List[DataPoint]
    selectedAlgorithm: str
    accuracy: float
    mae: float
    rmse: float
    allAlgorithms: Optional[List[AlgorithmResult]] = None

class MultiForecastResult(BaseModel):
    """Schema for multi-forecast result"""
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    totalCombinations: int

class SavedForecastRequest(BaseModel):
    """Schema for saving forecast result"""
    name: str
    description: Optional[str] = None
    forecast_config: dict
    forecast_data: dict

class SavedForecastResponse(BaseModel):
    """Schema for saved forecast response"""
    id: int
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class BestFitRecommendationRequest(BaseModel):
    """Schema for best fit recommendation request"""
    config: ForecastConfig

class BestFitRecommendationResponse(BaseModel):
    """Schema for best fit recommendation response"""
    recommended_algorithm: Optional[str]
    confidence: Optional[float] = None
    last_accuracy: Optional[float] = None
    last_run_date: Optional[str] = None
    message: str
    historical_performance: Optional[List[Dict[str, Any]]] = None
