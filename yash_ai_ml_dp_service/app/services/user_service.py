#!/usr/bin/env python3
"""
User management service
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from app.repositories.user_repository import (
    get_user_by_username,
    get_user_by_email,
    create_user,
    get_all_users,
    update_user_status,
    update_user_approval,
    delete_user
)
from app.models.user import User

def register_new_user(
    db: Session,
    username: str,
    email: str,
    password: str,
    full_name: Optional[str] = None
) -> dict:
    """Register a new user"""
    # Check if username exists
    if get_user_by_username(db, username):
        return {"success": False, "message": "Username already exists"}
    
    # Check if email exists
    if get_user_by_email(db, email):
        return {"success": False, "message": "Email already registered"}
    
    # Create user
    user = create_user(db, username, email, password, full_name)
    
    return {
        "success": True,
        "message": "User registered successfully. Please wait for admin approval.",
        "user": user
    }

def approve_user(db: Session, user_id: int) -> dict:
    """Approve a user"""
    user = update_user_approval(db, user_id, True)
    if user:
        return {"success": True, "message": "User approved successfully", "user": user}
    return {"success": False, "message": "User not found"}

def deactivate_user(db: Session, user_id: int) -> dict:
    """Deactivate a user"""
    user = update_user_status(db, user_id, False)
    if user:
        return {"success": True, "message": "User deactivated successfully", "user": user}
    return {"success": False, "message": "User not found"}

def activate_user(db: Session, user_id: int) -> dict:
    """Activate a user"""
    user = update_user_status(db, user_id, True)
    if user:
        return {"success": True, "message": "User activated successfully", "user": user}
    return {"success": False, "message": "User not found"}

def get_users_for_admin(db: Session) -> List[User]:
    """Get all users for admin view"""
    return get_all_users(db)

def remove_user(db: Session, user_id: int) -> dict:
    """Remove a user"""
    if delete_user(db, user_id):
        return {"success": True, "message": "User deleted successfully"}
    return {"success": False, "message": "User not found"}

def create_default_admin_user(db: Session) -> Optional[User]:
    """Create default admin user if none exists"""
    from app.repositories.user_repository import get_user_count
    
    if get_user_count(db) == 0:
        user = create_user(
            db,
            username="admin",
            email="admin@forecasting.com",
            password="admin123",
            full_name="System Administrator"
        )
        user.is_admin = True
        user.is_approved = True
        user.is_active = True
        db.commit()
        return user
    return None
