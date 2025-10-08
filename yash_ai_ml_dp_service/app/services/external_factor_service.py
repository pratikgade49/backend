#!/usr/bin/env python3
"""
External factor service - handles FRED API integration and external data
"""

import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.repositories.external_factor_repository import (
    get_all_external_factors,
    get_external_factor_by_name,
    create_external_factor,
    get_external_factors_for_date_range
)
from app.models.external_factor import ExternalFactorData

class ExternalFactorService:
    """Service for managing external factors and FRED API integration"""

    FRED_API_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    FRED_API_KEY = os.getenv("FRED_API_KEY", "82a8e6191d71f41b22cf33bf73f7a0c2")

    @staticmethod
    def get_available_factors(db: Session) -> List[str]:
        """Get list of available external factor names"""
        factors = get_all_external_factors(db)
        return list(set([f.factor_name for f in factors]))

    @staticmethod
    def fetch_from_fred(series_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch data from FRED API"""
        params = {
            "series_id": series_id,
            "api_key": ExternalFactorService.FRED_API_KEY,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date
        }

        try:
            response = requests.get(ExternalFactorService.FRED_API_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if "observations" in data:
                return [
                    {
                        "date": obs["date"],
                        "value": float(obs["value"]) if obs["value"] != "." else None
                    }
                    for obs in data["observations"]
                    if obs["value"] != "."
                ]
            return []
        except Exception as e:
            print(f"Error fetching from FRED: {e}")
            return []

    @staticmethod
    def import_fred_data(db: Session, series_id: str, factor_name: str, 
                        start_date: str, end_date: str) -> int:
        """Import data from FRED API into database"""
        observations = ExternalFactorService.fetch_from_fred(series_id, start_date, end_date)

        count = 0
        for obs in observations:
            if obs["value"] is not None:
                try:
                    create_external_factor(
                        db,
                        factor_name=factor_name,
                        date=datetime.strptime(obs["date"], "%Y-%m-%d").date(),
                        value=obs["value"]
                    )
                    count += 1
                except Exception as e:
                    continue

        return count

    @staticmethod
    def get_factors_for_forecast(db: Session, factor_names: List[str], 
                                 start_date: datetime, end_date: datetime) -> Dict[str, List[Dict]]:
        """Get external factors data for forecasting"""
        result = {}

        for factor_name in factor_names:
            factors = get_external_factors_for_date_range(
                db, factor_name, start_date.date(), end_date.date()
            )
            result[factor_name] = [
                {"date": f.date, "value": f.value}
                for f in factors
            ]

        return result

    @staticmethod
    def create_factor(db: Session, factor_name: str, date: datetime, value: float) -> ExternalFactorData:
        """Create a new external factor entry"""
        return create_external_factor(db, factor_name, date.date(), value)

    @staticmethod
    def get_factor_by_name(db: Session, factor_name: str) -> List[ExternalFactorData]:
        """Get all data for a specific factor"""
        return get_external_factor_by_name(db, factor_name)
