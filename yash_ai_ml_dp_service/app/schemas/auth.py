#!/usr/bin/env python3
"""
Authentication schemas
"""

from pydantic import BaseModel

class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str
    user: dict
