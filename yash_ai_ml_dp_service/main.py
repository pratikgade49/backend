#!/usr/bin/env python3
"""
Main FastAPI application with modular router structure
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import warnings

from app.core.config import settings
from app.core.database import init_database, engine
from app.api import (
    auth, users, data, forecast, admin, upload,
    configuration, saved_forecast, external_factor,
    model_cache, scheduler, database as db_router, downloads, root
)

from app.models.model_persistence import SavedModel, ModelAccuracyHistory
from app.core.scheduler_background import start_scheduler, stop_scheduler

warnings.filterwarnings('ignore')

app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup Event
@app.on_event("startup")
async def startup_event():
    print(f"🚀 Starting {settings.APP_TITLE}...")
    if init_database():
        print("✅ Database initialization successful!")
        try:
            SavedModel.metadata.create_all(bind=engine)
            ModelAccuracyHistory.metadata.create_all(bind=engine)
            print("✅ Model persistence tables initialized!")
        except Exception as e:
            print(f"⚠️  Model persistence table creation failed: {e}")

        # Create default admin user if none exists
        from app.core.database import SessionLocal

        from app.services.user_service import UserService
        db = SessionLocal()
        try:
            admin_user = UserService.create_default_admin_user(db)
            if admin_user:
                print("✅ Default admin user created: username='admin', password='admin123'")
            else:
                print("ℹ️  Admin user already exists")
        except Exception as e:
            print(f"⚠️  Error creating default admin user: {e}")
        finally:
            db.close()
    else:
        print("⚠️  Database initialization failed")
    
    # Start scheduler
    start_scheduler()
    print("✅ Forecast scheduler started")

# Shutdown Event
@app.on_event("shutdown")
async def shutdown_event():
    stop_scheduler()
    print("🛑 Forecast scheduler stopped")

# Register Routers
app.include_router(root.router)  # Root must be first for "/" endpoint
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(users.router)
app.include_router(upload.router)
app.include_router(data.router)
app.include_router(forecast.router)
app.include_router(configuration.router)
app.include_router(saved_forecast.router)
app.include_router(external_factor.router)
app.include_router(model_cache.router)
app.include_router(scheduler.router)
app.include_router(db_router.router)
app.include_router(downloads.router)
# Add these as you create them:
app.include_router(configuration.router)
app.include_router(saved_forecast.router)
app.include_router(external_factor.router)
app.include_router(model_cache.router)
app.include_router(scheduler.router)
app.include_router(db_router.router)
app.include_router(downloads.router)
app.include_router(root.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
