#!/usr/bin/env python3
"""
Configuration service - handles forecast configuration management
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from app.models.forecast import ForecastConfiguration
from app.models.user import User

class ConfigurationService:
    """Service for managing forecast configurations"""

    @staticmethod
    def create_configuration(db: Session, user_id: int, name: str, 
                           description: Optional[str], config_data: dict) -> ForecastConfiguration:
        """Create a new forecast configuration"""
        config = ForecastConfiguration(
            user_id=user_id,
            name=name,
            description=description,
            config=config_data
        )
        db.add(config)
        db.commit()
        db.refresh(config)
        return config

    @staticmethod
    def get_user_configurations(db: Session, user_id: int) -> List[ForecastConfiguration]:
        """Get all configurations for a user"""
        return db.query(ForecastConfiguration).filter(
            ForecastConfiguration.user_id == user_id
        ).order_by(ForecastConfiguration.updated_at.desc()).all()

    @staticmethod
    def get_configuration_by_id(db: Session, config_id: int, user_id: int) -> Optional[ForecastConfiguration]:
        """Get a specific configuration by ID"""
        return db.query(ForecastConfiguration).filter(
            ForecastConfiguration.id == config_id,
            ForecastConfiguration.user_id == user_id
        ).first()

    @staticmethod
    def update_configuration(db: Session, config_id: int, user_id: int,
                           name: Optional[str] = None,
                           description: Optional[str] = None,
                           config_data: Optional[dict] = None) -> Optional[ForecastConfiguration]:
        """Update an existing configuration"""
        config = ConfigurationService.get_configuration_by_id(db, config_id, user_id)
        
        if not config:
            return None

        if name is not None:
            config.name = name
        if description is not None:
            config.description = description
        if config_data is not None:
            config.config = config_data

        config.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(config)
        return config

    @staticmethod
    def delete_configuration(db: Session, config_id: int, user_id: int) -> bool:
        """Delete a configuration"""
        config = ConfigurationService.get_configuration_by_id(db, config_id, user_id)
        
        if not config:
            return False

        db.delete(config)
        db.commit()
        return True
