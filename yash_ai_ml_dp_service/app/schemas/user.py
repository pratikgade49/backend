#!/usr/bin/env python3
"""
User schemas for request/response validation
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    """Schema for user registration"""
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str

class UserResponse(BaseModel):
    """Schema for user response"""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_approved: bool
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True

class AdminSetActiveRequest(BaseModel):
    """Schema for admin to set user active status"""
    user_id: int
    is_active: bool
