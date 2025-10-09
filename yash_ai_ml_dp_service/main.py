#!/usr/bin/env python3
"""
Main FastAPI application with modular router structure
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import warnings

from app.core.config import settings
from app.core.database import init_database, engine
from app.api import auth, users, data, forecast, admin
from app.api import configuration, saved_forecasts, external_factors
from app.api import model_cache, scheduler, database as db_router, downloads, root
from app.models.model_persistence import SavedModel, ModelAccuracyHistory
from app.services.scheduler_service import start_scheduler, stop_scheduler

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
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(users.router)
app.include_router(data.router)
app.include_router(forecast.router)
# Add these as you create them:
app.include_router(configuration.router)
app.include_router(saved_forecasts.router)
app.include_router(external_factors.router)
app.include_router(model_cache.router)
app.include_router(scheduler.router)
app.include_router(db_router.router)
app.include_router(downloads.router)
app.include_router(root.router)

# Health Check (temporary - move to root.py later)
@app.get("/")
async def health_check():
    return {
        "message": f"{settings.APP_TITLE} is running",
        "version": settings.APP_VERSION
    }
