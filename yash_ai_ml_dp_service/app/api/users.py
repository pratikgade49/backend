#!/usr/bin/env python3
"""
User management API routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.core.security import get_current_user, require_admin
from app.schemas.user import UserResponse, AdminSetActiveRequest
from app.services.user_service import UserService
from app.models.user import User

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/", response_model=List[UserResponse], dependencies=[Depends(require_admin)])
async def get_all_users(db: Session = Depends(get_db)):
    """Get all users (admin only)"""
    users = UserService.get_all_users(db)
    
    return [
        UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_approved=user.is_approved,
            is_admin=user.is_admin,
            created_at=user.created_at.isoformat()
        )
        for user in users
    ]

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get user by ID"""
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this user"
        )
    
    user = UserService.get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_approved=user.is_approved,
        is_admin=user.is_admin,
        created_at=user.created_at.isoformat()
    )

@router.patch("/{user_id}/active", dependencies=[Depends(require_admin)])
async def set_user_active(
    user_id: int,
    request: AdminSetActiveRequest,
    db: Session = Depends(get_db)
):
    """Set user active status (admin only)"""
    user = UserService.set_user_active(db, user_id, request.is_active)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "message": f"User {'activated' if request.is_active else 'deactivated'} successfully",
        "user_id": user_id,
        "is_active": user.is_active
    }

@router.patch("/{user_id}/approve", dependencies=[Depends(require_admin)])
async def approve_user(user_id: int, db: Session = Depends(get_db)):
    """Approve a user (admin only)"""
    user = UserService.approve_user(db, user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "message": "User approved successfully",
        "user_id": user_id,
        "is_approved": user.is_approved
    }

@router.delete("/{user_id}", dependencies=[Depends(require_admin)])
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Delete a user (admin only)"""
    success = UserService.delete_user(db, user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User deleted successfully", "user_id": user_id}
