#!/usr/bin/env python3
"""
Database configuration and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Create engine
try:
    engine = create_engine(settings.DATABASE_URL, echo=False)
except Exception as e:
    print(f"❌ Error connecting to database: {e}")
    print("Please ensure:")
    print("1. PostgreSQL server is running")
    print("2. Database exists")
    print("3. Connection credentials are correct")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database tables"""
    try:
        # Test connection
        with engine.connect() as connection:
            print("✅ Database connection successful!")
        
        # Import all models to ensure they're registered
        from app.models import (
            user, dimension, forecast, external_factor, 
            scheduler, model_persistence
        )
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables verified/created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False
