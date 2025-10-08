#!/usr/bin/env python3
"""
Forecasting service - handles all ML algorithms and forecasting logic
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import warnings

from app.repositories.dimension_repository import DimensionManager
from app.repositories.forecast_repository import get_forecast_data_by_filters
from app.models.forecast import ForecastData

warnings.filterwarnings('ignore')

class ForecastingEngine:
    """Advanced forecasting engine with multiple algorithms"""

    ALGORITHMS = {
        "linear_regression": "Linear Regression",
        "polynomial_regression": "Polynomial Regression",
        "exponential_smoothing": "Exponential Smoothing",
        "holt_winters": "Holt-Winters",
        "arima": "ARIMA (Simple)",
        "random_forest": "Random Forest",
        "seasonal_decomposition": "Seasonal Decomposition",
        "moving_average": "Moving Average",
        "sarima": "SARIMA (Seasonal ARIMA)",
        "prophet_like": "Prophet-like Forecasting",
        "lstm_like": "Simple LSTM-like",
        "xgboost": "XGBoost Regression",
        "svr": "Support Vector Regression",
        "knn": "K-Nearest Neighbors",
        "gaussian_process": "Gaussian Process",
        "neural_network": "Neural Network (MLP)",
        "theta_method": "Theta Method",
        "croston": "Croston's Method",
        "ses": "Simple Exponential Smoothing",
        "damped_trend": "Damped Trend Method",
        "naive_seasonal": "Naive Seasonal",
        "drift_method": "Drift Method",
        "best_fit": "Best Fit (Auto-Select)",
        "best_statistical": "Best Statistical Method",
        "best_ml": "Best Machine Learning Method",
        "best_specialized": "Best Specialized Method"
    }

    @staticmethod
    def load_data_from_db(db: Session, config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from database based on forecast configuration"""
        product_ids = []
        customer_ids = []
        location_ids = []

        if config.get('multiSelect'):
            if config.get('selectedProducts'):
                product_ids = [DimensionManager.get_dimension_id(db, 'product', p) 
                             for p in config['selectedProducts'] 
                             if DimensionManager.get_dimension_id(db, 'product', p) is not None]
            if config.get('selectedCustomers'):
                customer_ids = [DimensionManager.get_dimension_id(db, 'customer', c) 
                              for c in config['selectedCustomers'] 
                              if DimensionManager.get_dimension_id(db, 'customer', c) is not None]
            if config.get('selectedLocations'):
                location_ids = [DimensionManager.get_dimension_id(db, 'location', l) 
                              for l in config['selectedLocations'] 
                              if DimensionManager.get_dimension_id(db, 'location', l) is not None]
        else:
            if config.get('selectedProduct'):
                pid = DimensionManager.get_dimension_id(db, 'product', config['selectedProduct'])
                if pid:
                    product_ids = [pid]
            if config.get('selectedCustomer'):
                cid = DimensionManager.get_dimension_id(db, 'customer', config['selectedCustomer'])
                if cid:
                    customer_ids = [cid]
            if config.get('selectedLocation'):
                lid = DimensionManager.get_dimension_id(db, 'location', config['selectedLocation'])
                if lid:
                    location_ids = [lid]

        records = get_forecast_data_by_filters(
            db,
            product_ids=product_ids if product_ids else None,
            customer_ids=customer_ids if customer_ids else None,
            location_ids=location_ids if location_ids else None
        )

        if not records:
            return pd.DataFrame()

        data_list = []
        for record in records:
            data_list.append({
                'date': record.date,
                'quantity': record.quantity,
                'product_id': record.product_id,
                'customer_id': record.customer_id,
                'location_id': record.location_id
            })

        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        return df

    @staticmethod
    def forecast_external_factors(data: pd.DataFrame, external_factor_cols: List[str], periods: int) -> Dict[str, np.ndarray]:
        """Forecast external factors using simple linear regression"""
        forecasts = {}
        
        for col in external_factor_cols:
            if col in data.columns:
                values = data[col].values
                if len(values) > 0:
                    X = np.arange(len(values)).reshape(-1, 1)
                    y = values
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    future_X = np.arange(len(values), len(values) + periods).reshape(-1, 1)
                    forecast = model.predict(future_X)
                    forecasts[col] = forecast
        
        return forecasts

    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        actual_non_zero = actual[actual != 0]
        predicted_non_zero = predicted[actual != 0]
        
        if len(actual_non_zero) > 0:
            mape = np.mean(np.abs((actual_non_zero - predicted_non_zero) / actual_non_zero)) * 100
            accuracy = 100 - mape
        else:
            accuracy = 0.0
        
        return {
            'accuracy': max(0, min(100, accuracy)),
            'mae': float(mae),
            'rmse': float(rmse)
        }

    @staticmethod
    def determine_trend(data: np.ndarray) -> str:
        """Determine trend direction"""
        if len(data) < 2:
            return "stable"
        
        X = np.arange(len(data)).reshape(-1, 1)
        y = data
        
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        
        if slope > data.std() * 0.1:
            return "increasing"
        elif slope < -data.std() * 0.1:
            return "decreasing"
        else:
            return "stable"

    @staticmethod
    def linear_regression_forecast(data: np.ndarray, periods: int, external_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Linear regression forecasting"""
        X = np.arange(len(data)).reshape(-1, 1)
        
        if external_data is not None and not external_data.empty:
            X = np.column_stack([X] + [external_data[col].values for col in external_data.columns])
        
        y = data
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
        
        if external_data is not None and not external_data.empty:
            future_external = ForecastingEngine.forecast_external_factors(
                external_data, list(external_data.columns), periods
            )
            external_features = np.column_stack([future_external[col] for col in external_data.columns])
            future_X = np.column_stack([future_X, external_features])
        
        forecast = model.predict(future_X)
        return np.maximum(forecast, 0)

    @staticmethod
    def polynomial_regression_forecast(data: np.ndarray, periods: int, degree: int = 2) -> np.ndarray:
        """Polynomial regression forecasting"""
        X = np.arange(len(data)).reshape(-1, 1)
        y = data
        
        X_poly = np.column_stack([X**i for i in range(1, degree + 1)])
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
        future_X_poly = np.column_stack([future_X**i for i in range(1, degree + 1)])
        
        forecast = model.predict(future_X_poly)
        return np.maximum(forecast, 0)

    @staticmethod
    def exponential_smoothing_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """Exponential smoothing forecasting"""
        try:
            model = ExponentialSmoothing(data, seasonal_periods=min(12, len(data)//2), 
                                        trend='add', seasonal='add')
            fitted = model.fit()
            forecast = fitted.forecast(periods)
            return np.maximum(forecast, 0)
        except:
            return ForecastingEngine.linear_regression_forecast(data, periods)

    @staticmethod
    def holt_winters_forecast(data: np.ndarray, periods: int, seasonal_periods: int = 12) -> np.ndarray:
        """Holt-Winters forecasting"""
        try:
            if len(data) < seasonal_periods * 2:
                seasonal_periods = max(4, len(data) // 2)
            
            model = ExponentialSmoothing(
                data,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add',
                initialization_method="estimated"
            )
            fitted = model.fit()
            forecast = fitted.forecast(periods)
            return np.maximum(forecast, 0)
        except:
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

    @staticmethod
    def arima_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """Simple ARIMA-like forecasting"""
        try:
            diff = np.diff(data)
            if len(diff) > 0:
                mean_diff = np.mean(diff)
                last_value = data[-1]
                forecast = np.array([last_value + mean_diff * (i + 1) for i in range(periods)])
            else:
                forecast = np.full(periods, data[-1])
            
            return np.maximum(forecast, 0)
        except:
            return ForecastingEngine.linear_regression_forecast(data, periods)

    @staticmethod
    def random_forest_forecast(data: np.ndarray, periods: int, external_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Random Forest forecasting"""
        try:
            X = np.arange(len(data)).reshape(-1, 1)
            
            if external_data is not None and not external_data.empty:
                X = np.column_stack([X] + [external_data[col].values for col in external_data.columns])
            
            y = data
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X, y)
            
            future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
            
            if external_data is not None and not external_data.empty:
                future_external = ForecastingEngine.forecast_external_factors(
                    external_data, list(external_data.columns), periods
                )
                external_features = np.column_stack([future_external[col] for col in external_data.columns])
                future_X = np.column_stack([future_X, external_features])
            
            forecast = model.predict(future_X)
            return np.maximum(forecast, 0)
        except Exception as e:
            return ForecastingEngine.linear_regression_forecast(data, periods, external_data)

    @staticmethod
    def seasonal_decomposition_forecast(data: np.ndarray, periods: int, seasonal_periods: int = 12) -> np.ndarray:
        """Seasonal decomposition forecasting"""
        try:
            if len(data) < seasonal_periods * 2:
                return ForecastingEngine.linear_regression_forecast(data, periods)
            
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            series = pd.Series(data)
            decomposition = seasonal_decompose(series, model='additive', period=seasonal_periods, extrapolate_trend='freq')
            
            trend = decomposition.trend.values
            seasonal = decomposition.seasonal.values
            
            trend_model = LinearRegression()
            X_trend = np.arange(len(trend)).reshape(-1, 1)
            trend_model.fit(X_trend, trend)
            
            future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
            future_trend = trend_model.predict(future_X)
            
            seasonal_pattern = seasonal[-seasonal_periods:]
            future_seasonal = np.tile(seasonal_pattern, (periods // seasonal_periods) + 1)[:periods]
            
            forecast = future_trend + future_seasonal
            return np.maximum(forecast, 0)
        except:
            return ForecastingEngine.holt_winters_forecast(data, periods, seasonal_periods)

    @staticmethod
    def moving_average_forecast(data: np.ndarray, periods: int, window: int = 3) -> np.ndarray:
        """Moving average forecasting"""
        if len(data) < window:
            window = len(data)
        
        forecast = []
        extended_data = list(data)
        
        for _ in range(periods):
            avg = np.mean(extended_data[-window:])
            forecast.append(avg)
            extended_data.append(avg)
        
        return np.maximum(np.array(forecast), 0)

    @staticmethod
    def sarima_forecast(data: np.ndarray, periods: int, seasonal_periods: int = 12) -> np.ndarray:
        """SARIMA forecasting"""
        return ForecastingEngine.holt_winters_forecast(data, periods, seasonal_periods)

    @staticmethod
    def prophet_like_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """Prophet-like forecasting"""
        return ForecastingEngine.seasonal_decomposition_forecast(data, periods)

    @staticmethod
    def lstm_like_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """LSTM-like forecasting using windowed approach"""
        try:
            window_size = min(5, len(data) - 1)
            if window_size < 1:
                return np.full(periods, data[-1])
            
            X, y = [], []
            for i in range(window_size, len(data)):
                X.append(data[i-window_size:i])
                y.append(data[i])
            
            X, y = np.array(X), np.array(y)
            
            model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
            model.fit(X, y)
            
            forecast = []
            current_window = list(data[-window_size:])
            
            for _ in range(periods):
                pred = model.predict([current_window])[0]
                forecast.append(pred)
                current_window = current_window[1:] + [pred]
            
            return np.maximum(np.array(forecast), 0)
        except:
            return ForecastingEngine.linear_regression_forecast(data, periods)

    @staticmethod
    def xgboost_forecast(data: np.ndarray, periods: int, external_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """XGBoost-like forecasting using Gradient Boosting"""
        try:
            X = np.arange(len(data)).reshape(-1, 1)
            
            if external_data is not None and not external_data.empty:
                X = np.column_stack([X] + [external_data[col].values for col in external_data.columns])
            
            y = data
            
            model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
            model.fit(X, y)
            
            future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
            
            if external_data is not None and not external_data.empty:
                future_external = ForecastingEngine.forecast_external_factors(
                    external_data, list(external_data.columns), periods
                )
                external_features = np.column_stack([future_external[col] for col in external_data.columns])
                future_X = np.column_stack([future_X, external_features])
            
            forecast = model.predict(future_X)
            return np.maximum(forecast, 0)
        except:
            return ForecastingEngine.random_forest_forecast(data, periods, external_data)

    @staticmethod
    def svr_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """Support Vector Regression forecasting"""
        try:
            X = np.arange(len(data)).reshape(-1, 1)
            y = data
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
            model.fit(X_scaled, y)
            
            future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
            future_X_scaled = scaler.transform(future_X)
            
            forecast = model.predict(future_X_scaled)
            return np.maximum(forecast, 0)
        except:
            return ForecastingEngine.linear_regression_forecast(data, periods)

    @staticmethod
    def knn_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """K-Nearest Neighbors forecasting"""
        try:
            X = np.arange(len(data)).reshape(-1, 1)
            y = data
            
            n_neighbors = min(5, len(data) - 1)
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            model.fit(X, y)
            
            future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
            forecast = model.predict(future_X)
            return np.maximum(forecast, 0)
        except:
            return ForecastingEngine.linear_regression_forecast(data, periods)

    @staticmethod
    def gaussian_process_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """Gaussian Process forecasting"""
        try:
            X = np.arange(len(data)).reshape(-1, 1)
            y = data
            
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
            model.fit(X, y)
            
            future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
            forecast, _ = model.predict(future_X, return_std=True)
            return np.maximum(forecast, 0)
        except:
            return ForecastingEngine.linear_regression_forecast(data, periods)

    @staticmethod
    def neural_network_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """Neural Network (MLP) forecasting"""
        try:
            X = np.arange(len(data)).reshape(-1, 1)
            y = data
            
            model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            model.fit(X, y)
            
            future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
            forecast = model.predict(future_X)
            return np.maximum(forecast, 0)
        except:
            return ForecastingEngine.linear_regression_forecast(data, periods)

    @staticmethod
    def theta_method_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """Theta Method forecasting"""
        try:
            theta = 2
            
            n = len(data)
            X = np.arange(n)
            
            b = np.polyfit(X, data, 1)[0]
            
            theta_line = data / theta + (1 - 1/theta) * (data[0] + b * X)
            
            alpha = 0.5
            ses_forecast = [theta_line[-1]]
            for _ in range(1, periods):
                ses_forecast.append(alpha * ses_forecast[-1] + (1 - alpha) * theta_line[-1])
            
            forecast = theta * np.array(ses_forecast) - (theta - 1) * (data[-1] + b * np.arange(1, periods + 1))
            
            return np.maximum(forecast, 0)
        except:
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

    @staticmethod
    def croston_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """Croston's Method for intermittent demand"""
        try:
            alpha = 0.1
            
            demand_sizes = []
            intervals = []
            
            last_demand_index = -1
            for i, value in enumerate(data):
                if value > 0:
                    demand_sizes.append(value)
                    if last_demand_index >= 0:
                        intervals.append(i - last_demand_index)
                    last_demand_index = i
            
            if not demand_sizes:
                return np.zeros(periods)
            
            avg_demand = demand_sizes[-1]
            avg_interval = intervals[-1] if intervals else 1
            
            for size in demand_sizes:
                avg_demand = alpha * size + (1 - alpha) * avg_demand
            
            for interval in intervals:
                avg_interval = alpha * interval + (1 - alpha) * avg_interval
            
            forecast_value = avg_demand / avg_interval if avg_interval > 0 else avg_demand
            
            return np.full(periods, max(0, forecast_value))
        except:
            return ForecastingEngine.moving_average_forecast(data, periods)

    @staticmethod
    def ses_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """Simple Exponential Smoothing"""
        try:
            alpha = 0.3
            
            smoothed = [data[0]]
            for i in range(1, len(data)):
                smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
            
            forecast = np.full(periods, smoothed[-1])
            return np.maximum(forecast, 0)
        except:
            return ForecastingEngine.moving_average_forecast(data, periods)

    @staticmethod
    def damped_trend_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """Damped Trend Method"""
        try:
            alpha = 0.3
            beta = 0.1
            phi = 0.9
            
            level = data[0]
            trend = data[1] - data[0] if len(data) > 1 else 0
            
            for value in data[1:]:
                prev_level = level
                level = alpha * value + (1 - alpha) * (level + phi * trend)
                trend = beta * (level - prev_level) + (1 - beta) * phi * trend
            
            forecast = []
            phi_sum = 0
            for i in range(1, periods + 1):
                phi_sum += phi ** i
                forecast.append(level + phi_sum * trend)
            
            return np.maximum(np.array(forecast), 0)
        except:
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

    @staticmethod
    def naive_seasonal_forecast(data: np.ndarray, periods: int, seasonal_periods: int = 12) -> np.ndarray:
        """Naive Seasonal forecasting"""
        if len(data) < seasonal_periods:
            return np.full(periods, data[-1])
        
        seasonal_pattern = data[-seasonal_periods:]
        forecast = np.tile(seasonal_pattern, (periods // seasonal_periods) + 1)[:periods]
        return np.maximum(forecast, 0)

    @staticmethod
    def drift_method_forecast(data: np.ndarray, periods: int) -> np.ndarray:
        """Drift Method forecasting"""
        if len(data) < 2:
            return np.full(periods, data[-1])
        
        drift = (data[-1] - data[0]) / (len(data) - 1)
        last_value = data[-1]
        
        forecast = np.array([last_value + drift * (i + 1) for i in range(periods)])
        return np.maximum(forecast, 0)

    @staticmethod
    def run_algorithm(algorithm: str, data: np.ndarray, periods: int, 
                     external_data: Optional[pd.DataFrame] = None,
                     seasonal_periods: int = 12) -> np.ndarray:
        """Run specific algorithm"""
        algorithm_map = {
            'linear_regression': lambda: ForecastingEngine.linear_regression_forecast(data, periods, external_data),
            'polynomial_regression': lambda: ForecastingEngine.polynomial_regression_forecast(data, periods),
            'exponential_smoothing': lambda: ForecastingEngine.exponential_smoothing_forecast(data, periods),
            'holt_winters': lambda: ForecastingEngine.holt_winters_forecast(data, periods, seasonal_periods),
            'arima': lambda: ForecastingEngine.arima_forecast(data, periods),
            'random_forest': lambda: ForecastingEngine.random_forest_forecast(data, periods, external_data),
            'seasonal_decomposition': lambda: ForecastingEngine.seasonal_decomposition_forecast(data, periods, seasonal_periods),
            'moving_average': lambda: ForecastingEngine.moving_average_forecast(data, periods),
            'sarima': lambda: ForecastingEngine.sarima_forecast(data, periods, seasonal_periods),
            'prophet_like': lambda: ForecastingEngine.prophet_like_forecast(data, periods),
            'lstm_like': lambda: ForecastingEngine.lstm_like_forecast(data, periods),
            'xgboost': lambda: ForecastingEngine.xgboost_forecast(data, periods, external_data),
            'svr': lambda: ForecastingEngine.svr_forecast(data, periods),
            'knn': lambda: ForecastingEngine.knn_forecast(data, periods),
            'gaussian_process': lambda: ForecastingEngine.gaussian_process_forecast(data, periods),
            'neural_network': lambda: ForecastingEngine.neural_network_forecast(data, periods),
            'theta_method': lambda: ForecastingEngine.theta_method_forecast(data, periods),
            'croston': lambda: ForecastingEngine.croston_forecast(data, periods),
            'ses': lambda: ForecastingEngine.ses_forecast(data, periods),
            'damped_trend': lambda: ForecastingEngine.damped_trend_forecast(data, periods),
            'naive_seasonal': lambda: ForecastingEngine.naive_seasonal_forecast(data, periods, seasonal_periods),
            'drift_method': lambda: ForecastingEngine.drift_method_forecast(data, periods),
        }
        
        if algorithm in algorithm_map:
            return algorithm_map[algorithm]()
        else:
            return ForecastingEngine.linear_regression_forecast(data, periods, external_data)