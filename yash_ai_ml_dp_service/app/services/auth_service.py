#!/usr/bin/env python3
"""
Authentication service
"""

from typing import Optional
from sqlalchemy.orm import Session
from datetime import timedelta
from app.core.security import create_access_token, verify_token
from app.repositories.user_repository import get_user_by_username
from app.models.user import User
from app.core.config import settings

class AuthService:
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        user = get_user_by_username(db, username)
        if not user:
            return None
        if not user.verify_password(password):
            return None
        return user

    @staticmethod
    def create_access_token(user: User) -> str:
        """Create access token for user"""
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=access_token_expires
        )
        return access_token

    @staticmethod
    def verify_user_token(token: str) -> Optional[str]:
        """Verify JWT token and return username"""
        return verify_token(token)

    @staticmethod
    def register_user(db: Session, username: str, email: str, password: str, full_name: Optional[str] = None):
        """Register a new user"""
        from app.services.user_service import register_new_user
        result = register_new_user(db, username, email, password, full_name)
        if result["success"]:
            return result["user"]
        else:
            raise ValueError(result["message"])
