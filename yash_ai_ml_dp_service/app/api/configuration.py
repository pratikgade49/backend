#!/usr/bin/env python3
"""
Configuration management API routes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.schemas.configuration import SaveConfigRequest, ConfigurationResponse
from app.services.configuration_service import ConfigurationService

router = APIRouter(prefix="/configurations", tags=["Configurations"])

@router.get("/", response_model=List[ConfigurationResponse])
async def get_configurations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all configurations for current user"""
    configs = ConfigurationService.get_user_configurations(db, current_user.id)
    return [
        ConfigurationResponse(
            id=config.id,
            name=config.name,
            config=config.config,
            created_at=config.created_at.isoformat()
        )
        for config in configs
    ]

@router.post("/", response_model=ConfigurationResponse)
async def save_configuration(
    request: SaveConfigRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Save a new configuration"""
    config = ConfigurationService.create_configuration(
        db=db,
        user_id=current_user.id,
        name=request.name,
        config=request.config
    )
    
    return ConfigurationResponse(
        id=config.id,
        name=config.name,
        config=config.config,
        created_at=config.created_at.isoformat()
    )

@router.get("/{config_id}", response_model=ConfigurationResponse)
async def get_configuration(
    config_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific configuration"""
    config = ConfigurationService.get_configuration(db, config_id, current_user.id)
    
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    return ConfigurationResponse(
        id=config.id,
        name=config.name,
        config=config.config,
        created_at=config.created_at.isoformat()
    )

@router.put("/{config_id}", response_model=ConfigurationResponse)
async def update_configuration(
    config_id: int,
    request: SaveConfigRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a configuration"""
    config = ConfigurationService.update_configuration(
        db=db,
        config_id=config_id,
        user_id=current_user.id,
        name=request.name,
        config_data=request.config
    )
    
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    return ConfigurationResponse(
        id=config.id,
        name=config.name,
        config=config.config,
        created_at=config.created_at.isoformat()
    )

@router.delete("/{config_id}")
async def delete_configuration(
    config_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a configuration"""
    success = ConfigurationService.delete_configuration(db, config_id, current_user.id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    return {"message": "Configuration deleted successfully"}
