#!/usr/bin/env python3
"""
Application configuration with environment variables
"""

import os
from typing import Optional
from pydantic import BaseModel

class Settings(BaseModel):
    """Application settings"""
    
    # Database Configuration
    DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "root")
    DB_NAME: str = os.getenv("DB_NAME", "forecasting_db")
    
    # Security Configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # External API Configuration
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "82a8e6191d71f41b22cf33bf73f7a0c2")
    FRED_API_BASE_URL: str = "https://api.stlouisfed.org/fred/series/observations"
    
    # Application Configuration
    APP_TITLE: str = "Multi-variant Forecasting API"
    APP_VERSION: str = "3.0.0"
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct database URL"""
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

# Global settings instance
settings = Settings()
