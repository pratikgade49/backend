#!/usr/bin/env python3
"""
Model persistence service for saving and loading trained forecasting models
"""

import pickle
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from app.models.model_persistence import SavedModel, ModelAccuracyHistory

class ModelPersistenceManager:
    """Manager for saving and loading trained models"""
    
    @staticmethod
    def generate_config_hash(config: Dict[str, Any]) -> str:
        """Generate a hash for the configuration"""
        if not config or not isinstance(config, dict):
            return "default_config_hash"
        
        def normalize_value(value):
            if isinstance(value, list):
                return sorted([str(v) for v in value])
            return str(value) if value is not None else ""
        
        def get_safe_list(value):
            if value is None:
                return []
            elif isinstance(value, list):
                return sorted(value)
            else:
                return [value]
        
        # Create normalized config for hashing
        normalized_config = {
            'forecastBy': config.get('forecastBy', ''),
            'selectedItem': normalize_value(config.get('selectedItem', '')),
            'selectedProduct': normalize_value(config.get('selectedProduct', '')),
            'selectedCustomer': normalize_value(config.get('selectedCustomer', '')),
            'selectedLocation': normalize_value(config.get('selectedLocation', '')),
            'algorithm': config.get('algorithm', ''),
            'interval': config.get('interval', ''),
            'historicPeriod': config.get('historicPeriod', 0),
            'forecastPeriod': config.get('forecastPeriod', 0),
            'selectedProducts': get_safe_list(config.get('selectedProducts')),
            'selectedCustomers': get_safe_list(config.get('selectedCustomers')),
            'selectedLocations': get_safe_list(config.get('selectedLocations')),
            'selectedItems': get_safe_list(config.get('selectedItems')),
            'multiSelect': config.get('multiSelect', False),
            'externalFactors': get_safe_list(config.get('externalFactors'))
        }
        
        config_str = json.dumps(normalized_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    @staticmethod
    def generate_data_hash(data: np.ndarray) -> str:
        """Generate a hash for the training data"""
        if data is None or not hasattr(data, 'tobytes'):
            return "default_data_hash"
        data_str = str(data.tobytes())
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def generate_model_hash(algorithm: str, config_hash: str, data_hash: str) -> str:
        """Generate a unique hash for the model"""
        combined = f"{algorithm}_{config_hash}_{data_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    @staticmethod
    def save_model(
        db: Session,
        model: Any,
        algorithm: str,
        config: Dict[str, Any],
        training_data: np.ndarray,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a trained model to the database"""
        try:
            # Generate hashes
            config_for_hash = config.copy()
            config_for_hash.pop('algorithm', None)
            config_hash = ModelPersistenceManager.generate_config_hash(config_for_hash)
            data_hash = ModelPersistenceManager.generate_data_hash(training_data)
            model_hash = ModelPersistenceManager.generate_model_hash(algorithm, config_hash, data_hash)
            
            # Check if model already exists
            existing_model = db.query(SavedModel).filter(SavedModel.model_hash == model_hash).first()
            
            if existing_model:
                # Update usage stats
                existing_model.last_used = datetime.utcnow()
                existing_model.use_count += 1
                db.commit()
                return model_hash
            
            # Serialize the model
            model_data = pickle.dumps(model)
            
            # Create new saved model record
            saved_model = SavedModel(
                model_hash=model_hash,
                algorithm=algorithm,
                config_hash=config_hash,
                model_data=model_data,
                model_metadata=json.dumps(metadata) if metadata else None,
                accuracy=metrics.get('accuracy'),
                mae=metrics.get('mae'),
                rmse=metrics.get('rmse'),
                use_count=1
            )
            
            db.add(saved_model)
            db.commit()
            
            return model_hash
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return None
    
    @staticmethod
    def load_model(db: Session, model_hash: str) -> Optional[Any]:
        """Load a trained model from the database"""
        try:
            saved_model = db.query(SavedModel).filter(SavedModel.model_hash == model_hash).first()
            if not saved_model:
                return None
            
            # Update usage stats
            saved_model.last_used = datetime.utcnow()
            saved_model.use_count += 1
            db.commit()
            
            # Deserialize the model
            model = pickle.loads(saved_model.model_data)
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    @staticmethod
    def find_cached_model(
        db: Session,
        algorithm: str,
        config: Dict[str, Any],
        training_data: np.ndarray
    ) -> Optional[str]:
        """Find if a cached model exists"""
        try:
            config_hash = ModelPersistenceManager.generate_config_hash(config)
            data_hash = ModelPersistenceManager.generate_data_hash(training_data)
            model_hash = ModelPersistenceManager.generate_model_hash(algorithm, config_hash, data_hash)
            
            saved_model = db.query(SavedModel).filter(SavedModel.model_hash == model_hash).first()
            
            return model_hash if saved_model else None
            
        except Exception as e:
            print(f"Error finding cached model: {e}")
            return None
    
    @staticmethod
    def record_accuracy_history(
        db: Session,
        model_hash: str,
        algorithm: str,
        config_hash: str,
        actual_values: List[float],
        predicted_values: List[float],
        metrics: Dict[str, float]
    ):
        """Record forecast accuracy for historical tracking"""
        try:
            history_record = ModelAccuracyHistory(
                model_hash=model_hash,
                algorithm=algorithm,
                config_hash=config_hash,
                forecast_date=datetime.utcnow(),
                actual_values=json.dumps(actual_values),
                predicted_values=json.dumps(predicted_values),
                accuracy=metrics.get('accuracy'),
                mae=metrics.get('mae'),
                rmse=metrics.get('rmse')
            )
            
            db.add(history_record)
            db.commit()
            
        except Exception as e:
            print(f"Error recording accuracy history: {e}")
    
    @staticmethod
    def get_accuracy_history(
        db: Session,
        config_hash: str,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get accuracy history for a configuration"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            history = db.query(ModelAccuracyHistory).filter(
                ModelAccuracyHistory.config_hash == config_hash,
                ModelAccuracyHistory.created_at >= cutoff_date
            ).order_by(ModelAccuracyHistory.created_at.desc()).all()
            
            result = []
            for record in history:
                result.append({
                    'algorithm': record.algorithm,
                    'forecast_date': record.forecast_date.isoformat(),
                    'accuracy': record.accuracy,
                    'mae': record.mae,
                    'rmse': record.rmse,
                    'actual_values': json.loads(record.actual_values) if record.actual_values else [],
                    'predicted_values': json.loads(record.predicted_values) if record.predicted_values else []
                })
            
            return result
            
        except Exception as e:
            print(f"Error getting accuracy history: {e}")
            return []
    
    @staticmethod
    def cleanup_old_models(db: Session, days_old: int = 30, max_models: int = 100):
        """Clean up old unused models"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Delete models that haven't been used recently
            old_models = db.query(SavedModel).filter(
                SavedModel.last_used < cutoff_date
            ).order_by(SavedModel.last_used.asc()).all()
            
            # Keep only the most recent models if we exceed max_models
            all_models = db.query(SavedModel).order_by(SavedModel.last_used.desc()).all()
            if len(all_models) > max_models:
                models_to_delete = all_models[max_models:]
                for model in models_to_delete:
                    db.delete(model)
            
            # Delete old models
            for model in old_models:
                db.delete(model)
            
            db.commit()
            
        except Exception as e:
            print(f"Error cleaning up old models: {e}")
