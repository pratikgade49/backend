#!/usr/bin/env python3
"""
Model persistence repository
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.models.model_persistence import SavedModel, ModelAccuracyHistory

def save_model_to_db(
    db: Session,
    model_hash: str,
    algorithm: str,
    config_hash: str,
    model_data: bytes,
    model_metadata: Optional[str] = None,
    accuracy: Optional[float] = None,
    mae: Optional[float] = None,
    rmse: Optional[float] = None
) -> SavedModel:
    """Save trained model to database"""
    saved_model = SavedModel(
        model_hash=model_hash,
        algorithm=algorithm,
        config_hash=config_hash,
        model_data=model_data,
        model_metadata=model_metadata,
        accuracy=accuracy,
        mae=mae,
        rmse=rmse
    )
    db.add(saved_model)
    db.commit()
    db.refresh(saved_model)
    return saved_model

def load_model_from_db(db: Session, model_hash: str) -> Optional[SavedModel]:
    """Load model from database by hash"""
    model = db.query(SavedModel).filter(SavedModel.model_hash == model_hash).first()
    
    if model:
        # Update usage statistics
        model.last_used = datetime.utcnow()
        model.use_count += 1
        db.commit()
    
    return model

def find_model_by_hash(db: Session, model_hash: str) -> Optional[SavedModel]:
    """Find model by exact hash match"""
    return db.query(SavedModel).filter(SavedModel.model_hash == model_hash).first()

def get_models_by_config_hash(db: Session, config_hash: str) -> List[SavedModel]:
    """Get all models for a specific configuration"""
    return db.query(SavedModel).filter(
        SavedModel.config_hash == config_hash
    ).order_by(SavedModel.last_used.desc()).all()

def get_best_model_for_config(
    db: Session,
    config_hash: str,
    algorithm: Optional[str] = None
) -> Optional[SavedModel]:
    """Get best performing model for a configuration"""
    query = db.query(SavedModel).filter(SavedModel.config_hash == config_hash)
    
    if algorithm:
        query = query.filter(SavedModel.algorithm == algorithm)
    
    return query.order_by(SavedModel.accuracy.desc()).first()

def cleanup_old_models(db: Session, days: int = 30) -> int:
    """Clean up old unused models"""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    old_models = db.query(SavedModel).filter(
        SavedModel.last_used < cutoff_date,
        SavedModel.use_count < 5
    ).all()
    
    count = len(old_models)
    for model in old_models:
        db.delete(model)
    
    db.commit()
    return count

def save_accuracy_history(
    db: Session,
    model_hash: str,
    algorithm: str,
    config_hash: str,
    forecast_date: datetime,
    actual_values: Optional[str] = None,
    predicted_values: Optional[str] = None,
    accuracy: Optional[float] = None,
    mae: Optional[float] = None,
    rmse: Optional[float] = None
) -> ModelAccuracyHistory:
    """Save model accuracy history"""
    history = ModelAccuracyHistory(
        model_hash=model_hash,
        algorithm=algorithm,
        config_hash=config_hash,
        forecast_date=forecast_date,
        actual_values=actual_values,
        predicted_values=predicted_values,
        accuracy=accuracy,
        mae=mae,
        rmse=rmse
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return history

def get_accuracy_history(
    db: Session,
    model_hash: Optional[str] = None,
    config_hash: Optional[str] = None,
    algorithm: Optional[str] = None,
    limit: int = 100
) -> List[ModelAccuracyHistory]:
    """Get accuracy history with filters"""
    query = db.query(ModelAccuracyHistory)
    
    if model_hash:
        query = query.filter(ModelAccuracyHistory.model_hash == model_hash)
    if config_hash:
        query = query.filter(ModelAccuracyHistory.config_hash == config_hash)
    if algorithm:
        query = query.filter(ModelAccuracyHistory.algorithm == algorithm)
    
    return query.order_by(ModelAccuracyHistory.forecast_date.desc()).limit(limit).all()

def get_model_count(db: Session) -> int:
    """Get total saved model count"""
    return db.query(SavedModel).count()
