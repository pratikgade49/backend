#!/usr/bin/env python3
"""
Hash generation utilities
"""

import hashlib

def generate_combination_key(*args) -> str:
    """Generate a stable hash key for combinations"""
    combined = "_".join(str(arg) for arg in sorted(args))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
