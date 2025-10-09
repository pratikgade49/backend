#!/usr/bin/env python3
"""
Hash generation utilities
"""

import hashlib
from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def generate_combination_key(*args) -> str:
    """Generate a stable hash key for combinations"""
    combined = "_".join(str(arg) for arg in sorted(args))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]

def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)
