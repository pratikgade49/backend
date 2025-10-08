#!/usr/bin/env python3
"""
User repository for database operations
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.user import User

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()

def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()

def create_user(db: Session, username: str, email: str, password: str, full_name: Optional[str] = None) -> User:
    """Create a new user"""
    user = User(
        username=username,
        email=email,
        hashed_password=User.get_password_hash(password),
        full_name=full_name,
        is_active=True,
        is_approved=False,
        is_admin=False
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_all_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    """Get all users"""
    return db.query(User).offset(skip).limit(limit).all()

def update_user_status(db: Session, user_id: int, is_active: bool) -> Optional[User]:
    """Update user active status"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.is_active = is_active
        db.commit()
        db.refresh(user)
    return user

def update_user_approval(db: Session, user_id: int, is_approved: bool) -> Optional[User]:
    """Update user approval status"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.is_approved = is_approved
        db.commit()
        db.refresh(user)
    return user

def delete_user(db: Session, user_id: int) -> bool:
    """Delete a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        db.delete(user)
        db.commit()
        return True
    return False

def get_user_count(db: Session) -> int:
    """Get total user count"""
    return db.query(User).count()
