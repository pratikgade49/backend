#!/usr/bin/env python3
"""
Application startup and shutdown event handlers
"""

import logging
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, init_database
from app.models.user import User

logger = logging.getLogger(__name__)

def create_default_admin_user():
    """Create default admin user if no users exist"""
    try:
        db = SessionLocal()
        user_count = db.query(User).count()
        if user_count == 0:
            default_user = User(
                username="admin",
                email="admin@forecasting.com",
                hashed_password=User.get_password_hash("admin123"),
                full_name="System Administrator",
                is_active=True,
                is_approved=True,
                is_admin=True
            )
            db.add(default_user)
            db.commit()
            logger.info("✅ Default admin user created (username: admin, password: admin123)")
        db.close()
    except Exception as e:
        logger.error(f"⚠️ Error creating default user: {e}")

async def startup_handler():
    """Handle application startup"""
    logger.info("🚀 Starting Multi-variant Forecasting API...")
    
    if init_database():
        logger.info("✅ Database initialization successful!")
        create_default_admin_user()
    else:
        logger.warning("⚠️ Database initialization failed - some features may not work")
    
    # Start scheduler
    from app.services.scheduler_service import start_scheduler
    start_scheduler()
    logger.info("✅ Forecast scheduler started")

async def shutdown_handler():
    """Handle application shutdown"""
    logger.info("🛑 Shutting down application...")
    
    # Stop scheduler
    from app.services.scheduler_service import stop_scheduler
    stop_scheduler()
    logger.info("✅ Forecast scheduler stopped")
