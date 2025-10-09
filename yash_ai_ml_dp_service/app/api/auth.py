#!/usr/bin/env python3
"""
Authentication API routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user, get_current_user_optional
from app.schemas import UserLogin, Token
from app.schemas.user import UserCreate, UserResponse
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        user = AuthService.register_user(
            db=db,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login and get access token"""
    user = AuthService.authenticate_user(db, credentials.username, credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    if not user.is_approved:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is pending approval"
        )
    
    access_token = AuthService.create_access_token(user)

    return Token(
        access_token=access_token,
        token_type="bearer",
        user={
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_admin": user.is_admin
        }
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_approved=current_user.is_approved,
        is_admin=current_user.is_admin,
        created_at=current_user.created_at.isoformat()
    )

@router.get("/check")
async def check_auth(current_user: User = Depends(get_current_user_optional)):
    """Check if user is authenticated"""
    if current_user:
        return {
            "authenticated": True,
            "user": {
                "id": current_user.id,
                "username": current_user.username,
                "email": current_user.email,
                "is_admin": current_user.is_admin
            }
        }
    return {"authenticated": False}