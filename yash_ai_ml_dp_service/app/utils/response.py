#!/usr/bin/env python3
"""
Standardized API response builders
"""

from typing import Any, Dict, Optional

def success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Build a success response"""
    return {
        "success": True,
        "message": message,
        "data": data
    }

def error_response(message: str, details: Optional[Any] = None) -> Dict[str, Any]:
    """Build an error response"""
    response = {
        "success": False,
        "message": message
    }
    if details:
        response["details"] = details
    return response

def paginated_response(
    data: Any,
    total: int,
    page: int = 1,
    page_size: int = 100
) -> Dict[str, Any]:
    """Build a paginated response"""
    return {
        "success": True,
        "data": data,
        "pagination": {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
    }
