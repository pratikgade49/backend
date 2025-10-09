#!/usr/bin/env python3
"""
Configuration schemas
"""

from pydantic import BaseModel
from typing import Optional

class SaveConfigRequest(BaseModel):
    """Schema for saving forecast configuration"""
    name: str
    config: dict

class ConfigurationResponse(BaseModel):
    """Schema for configuration response"""
    id: int
    name: str
    description: Optional[str]
    config: dict
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True
