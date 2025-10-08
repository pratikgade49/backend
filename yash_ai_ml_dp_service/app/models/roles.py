#!/usr/bin/env python3
"""
User roles and permissions
"""

from enum import Enum as PyEnum

class UserRole(PyEnum):
    """User role enumeration"""
    ADMIN = "admin"
    USER = "user"
