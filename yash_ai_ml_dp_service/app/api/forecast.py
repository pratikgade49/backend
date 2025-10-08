#!/usr/bin/env python3
"""
Forecasting API routes
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import hashlib
import json

from app.core.database import get_db
from app.core.security import get_current_user, get_current_user_optional
from app.schemas.forecast import (
    ForecastConfig,
    ForecastResult,
    MultiForecastResult,
    AlgorithmResult,
    DataPoint,
    BestFitRecommendationRequest,
    BestFitRecommendationResponse
)
from app.services.forecasting_service import ForecastingEngine
from app.models.user import User
from app.utils.response import create_forecast_response
import pandas as pd
import numpy as np

router = APIRouter(prefix="/forecast", tags=["Forecasting"])

def generate_config_hash(config: dict) -> str:
    """Generate a hash for configuration caching"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

@router.post("/generate", response_model=ForecastResult)
async def generate_forecast(
    config: ForecastConfig,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate a forecast based on configuration"""
    try:
        config_dict = config.dict()
        config_hash = generate_config_hash(config_dict)
        
        df = ForecastingEngine.load_data_from_db(db, config_dict)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the specified configuration")
        
        aggregated = df.groupby('date')['quantity'].sum().reset_index()
        aggregated = aggregated.sort_values('date')
        
        data = aggregated['quantity'].values
        dates = pd.to_datetime(aggregated['date'])
        
        if len(data) < 2:
            raise HTTPException(status_code=400, detail="Not enough data points for forecasting")
        
        algorithm = config_dict.get('algorithm', 'linear_regression')
        forecast_periods = config_dict.get('forecastPeriod', 6)
        interval = config_dict.get('interval', 'month')
        
        seasonal_periods = 12 if interval == 'month' else 4 if interval == 'quarter' else 52
        
        forecast = ForecastingEngine.run_algorithm(
            algorithm=algorithm,
            data=data,
            periods=forecast_periods,
            seasonal_periods=seasonal_periods
        )
        
        train_pred = ForecastingEngine.run_algorithm(
            algorithm=algorithm,
            data=data[:-1],
            periods=1,
            seasonal_periods=seasonal_periods
        )
        
        if len(train_pred) > 0 and len(data) > 0:
            metrics = ForecastingEngine.calculate_metrics(
                actual=data[-len(train_pred):],
                predicted=train_pred
            )
        else:
            metrics = {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0}
        
        trend = ForecastingEngine.determine_trend(data)
        
        last_date = dates.iloc[-1]
        if interval == 'month':
            future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq='MS')[1:]
        elif interval == 'quarter':
            future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq='QS')[1:]
        else:
            future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq='W')[1:]
        
        historic_data = [
            DataPoint(
                date=dates.iloc[i].strftime('%Y-%m-%d'),
                quantity=float(data[i]),
                period=dates.iloc[i].strftime('%Y-%m')
            )
            for i in range(len(data))
        ]
        
        forecast_data = [
            DataPoint(
                date=future_dates[i].strftime('%Y-%m-%d'),
                quantity=float(forecast[i]),
                period=future_dates[i].strftime('%Y-%m')
            )
            for i in range(len(forecast))
        ]
        
        return ForecastResult(
            selectedAlgorithm=algorithm,
            accuracy=metrics['accuracy'],
            mae=metrics['mae'],
            rmse=metrics['rmse'],
            historicData=historic_data,
            forecastData=forecast_data,
            trend=trend,
            configHash=config_hash
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")

@router.post("/compare-algorithms")
async def compare_algorithms(
    config: ForecastConfig,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Compare multiple algorithms for the same dataset"""
    try:
        config_dict = config.dict()
        
        df = ForecastingEngine.load_data_from_db(db, config_dict)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        aggregated = df.groupby('date')['quantity'].sum().reset_index()
        aggregated = aggregated.sort_values('date')
        
        data = aggregated['quantity'].values
        dates = pd.to_datetime(aggregated['date'])
        
        if len(data) < 2:
            raise HTTPException(status_code=400, detail="Not enough data points")
        
        forecast_periods = config_dict.get('forecastPeriod', 6)
        interval = config_dict.get('interval', 'month')
        seasonal_periods = 12 if interval == 'month' else 4 if interval == 'quarter' else 52
        
        algorithms_to_test = [
            'linear_regression', 'polynomial_regression', 'exponential_smoothing',
            'holt_winters', 'arima', 'random_forest', 'seasonal_decomposition',
            'moving_average', 'xgboost', 'svr', 'knn', 'neural_network'
        ]
        
        results = []
        
        for algo in algorithms_to_test:
            try:
                forecast = ForecastingEngine.run_algorithm(
                    algorithm=algo,
                    data=data,
                    periods=forecast_periods,
                    seasonal_periods=seasonal_periods
                )
                
                train_pred = ForecastingEngine.run_algorithm(
                    algorithm=algo,
                    data=data[:-1],
                    periods=1,
                    seasonal_periods=seasonal_periods
                )
                
                if len(train_pred) > 0:
                    metrics = ForecastingEngine.calculate_metrics(
                        actual=data[-len(train_pred):],
                        predicted=train_pred
                    )
                else:
                    metrics = {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0}
                
                trend = ForecastingEngine.determine_trend(data)
                
                last_date = dates.iloc[-1]
                if interval == 'month':
                    future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq='MS')[1:]
                elif interval == 'quarter':
                    future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq='QS')[1:]
                else:
                    future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq='W')[1:]
                
                historic_data = [
                    DataPoint(
                        date=dates.iloc[i].strftime('%Y-%m-%d'),
                        quantity=float(data[i]),
                        period=dates.iloc[i].strftime('%Y-%m')
                    )
                    for i in range(len(data))
                ]
                
                forecast_data = [
                    DataPoint(
                        date=future_dates[i].strftime('%Y-%m-%d'),
                        quantity=float(forecast[i]),
                        period=future_dates[i].strftime('%Y-%m')
                    )
                    for i in range(len(forecast))
                ]
                
                results.append(AlgorithmResult(
                    algorithm=algo,
                    accuracy=metrics['accuracy'],
                    mae=metrics['mae'],
                    rmse=metrics['rmse'],
                    historicData=historic_data,
                    forecastData=forecast_data,
                    trend=trend
                ))
            
            except Exception as e:
                continue
        
        results.sort(key=lambda x: x.accuracy, reverse=True)
        
        return {
            "algorithms": results,
            "best_algorithm": results[0].algorithm if results else None,
            "total_tested": len(results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

@router.post("/best-fit", response_model=BestFitRecommendationResponse)
async def recommend_best_fit(
    request: BestFitRecommendationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Recommend the best algorithm for the given configuration"""
    try:
        comparison_result = await compare_algorithms(request.config, db, current_user)
        
        if comparison_result.get('algorithms'):
            best = comparison_result['algorithms'][0]
            return BestFitRecommendationResponse(
                recommended_algorithm=best.algorithm,
                accuracy=best.accuracy,
                mae=best.mae,
                rmse=best.rmse,
                reason=f"Best accuracy ({best.accuracy:.2f}%) among {comparison_result['total_tested']} algorithms tested"
            )
        else:
            return BestFitRecommendationResponse(
                recommended_algorithm="linear_regression",
                reason="Default algorithm (no sufficient data for comparison)"
            )
    
    except Exception as e:
        return BestFitRecommendationResponse(
            recommended_algorithm="linear_regression",
            reason=f"Error during recommendation: {str(e)}"
        )