#!/usr/bin/env python3
"""
Admin API routes
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.core.security import require_admin
from app.schemas.user import UserResponse, AdminSetActiveRequest
from app.services.user_service import UserService

router = APIRouter(prefix="/admin", tags=["Admin"])

@router.get("/users", response_model=List[UserResponse])
async def get_all_users(
    db: Session = Depends(get_db),
    _: None = Depends(require_admin)
):
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
@router.post("/users/{user_id}/active", response_model=UserResponse)
async def set_user_active(
    user_id: int,
    request: AdminSetActiveRequest,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin)
):
    """Set user active status (admin only)"""
    user = UserService.set_user_active(db, user_id, request.is_active)
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
        created_at=user.created_at
    )


@router.patch("/users/{user_id}/approve")
async def approve_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin)
):
    """Approve a user (admin only)"""
    user = UserService.approve_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "message": "User approved successfully",
        "user_id": user_id,
        "is_approved": user.is_approved
    }

@router.patch("/users/{user_id}/active")
async def set_user_active(
    user_id: int,
    request: AdminSetActiveRequest,
    db: Session = Depends(get_db),
    _: None = Depends(require_admin)
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
