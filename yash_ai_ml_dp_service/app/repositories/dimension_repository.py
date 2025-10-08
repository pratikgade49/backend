#!/usr/bin/env python3
"""
Dimension repository for surrogate key operations
"""

from typing import Dict, Optional
from sqlalchemy.orm import Session
from app.models.dimension import (
    ProductDimension,
    CustomerDimension,
    LocationDimension,
    ProductCustomerCombination,
    ProductLocationCombination,
    CustomerLocationCombination,
    ProductCustomerLocationCombination
)
from app.utils.hash_utils import generate_combination_key

class DimensionManager:
    """Manager class for handling dimension and combination operations"""
    
    # Class level caches
    _dimension_cache = {
        'product': {},
        'customer': {},
        'location': {}
    }
    
    _combination_cache = {
        'product_customer': {},
        'product_location': {},
        'customer_location': {},
        'product_customer_location': {}
    }
    
    @staticmethod
    def get_or_create_dimension_cached(
        db: Session,
        dimension_type: str,
        name: str,
        **extra_fields
    ) -> Optional[int]:
        """Get or create dimension with caching"""
        if not name:
            return None
            
        # Check cache first
        if name in DimensionManager._dimension_cache[dimension_type]:
            return DimensionManager._dimension_cache[dimension_type][name]
        
        # Not in cache, query database
        if dimension_type == 'product':
            record = db.query(ProductDimension).filter(
                ProductDimension.product_name == name
            ).first()
            if not record:
                record = ProductDimension(product_name=name, **extra_fields)
                db.add(record)
                db.flush()
        elif dimension_type == 'customer':
            record = db.query(CustomerDimension).filter(
                CustomerDimension.customer_name == name
            ).first()
            if not record:
                record = CustomerDimension(customer_name=name, **extra_fields)
                db.add(record)
                db.flush()
        elif dimension_type == 'location':
            record = db.query(LocationDimension).filter(
                LocationDimension.location_name == name
            ).first()
            if not record:
                record = LocationDimension(location_name=name, **extra_fields)
                db.add(record)
                db.flush()
        
        # Cache the result
        DimensionManager._dimension_cache[dimension_type][name] = record.id
        return record.id
    
    @staticmethod
    def get_combination_ids(
        db: Session,
        product_id: Optional[int] = None,
        customer_id: Optional[int] = None,
        location_id: Optional[int] = None
    ) -> Dict[str, int]:
        """Get all combination IDs efficiently using caching"""
        result = {}
        
        # Product-Customer combination
        if product_id and customer_id:
            combo_key = f"{product_id}_{customer_id}"
            if combo_key in DimensionManager._combination_cache['product_customer']:
                result['product_customer_id'] = DimensionManager._combination_cache['product_customer'][combo_key]
            else:
                combo_key_hash = generate_combination_key(product_id, customer_id)
                record = db.query(ProductCustomerCombination).filter(
                    ProductCustomerCombination.combination_key == combo_key_hash
                ).first()
                if not record:
                    record = ProductCustomerCombination(
                        product_id=product_id,
                        customer_id=customer_id,
                        combination_key=combo_key_hash
                    )
                    db.add(record)
                    db.flush()
                DimensionManager._combination_cache['product_customer'][combo_key] = record.id
                result['product_customer_id'] = record.id
        
        # Product-Location combination
        if product_id and location_id:
            combo_key = f"{product_id}_{location_id}"
            if combo_key in DimensionManager._combination_cache['product_location']:
                result['product_location_id'] = DimensionManager._combination_cache['product_location'][combo_key]
            else:
                combo_key_hash = generate_combination_key(product_id, location_id)
                record = db.query(ProductLocationCombination).filter(
                    ProductLocationCombination.combination_key == combo_key_hash
                ).first()
                if not record:
                    record = ProductLocationCombination(
                        product_id=product_id,
                        location_id=location_id,
                        combination_key=combo_key_hash
                    )
                    db.add(record)
                    db.flush()
                DimensionManager._combination_cache['product_location'][combo_key] = record.id
                result['product_location_id'] = record.id
        
        # Customer-Location combination
        if customer_id and location_id:
            combo_key = f"{customer_id}_{location_id}"
            if combo_key in DimensionManager._combination_cache['customer_location']:
                result['customer_location_id'] = DimensionManager._combination_cache['customer_location'][combo_key]
            else:
                combo_key_hash = generate_combination_key(customer_id, location_id)
                record = db.query(CustomerLocationCombination).filter(
                    CustomerLocationCombination.combination_key == combo_key_hash
                ).first()
                if not record:
                    record = CustomerLocationCombination(
                        customer_id=customer_id,
                        location_id=location_id,
                        combination_key=combo_key_hash
                    )
                    db.add(record)
                    db.flush()
                DimensionManager._combination_cache['customer_location'][combo_key] = record.id
                result['customer_location_id'] = record.id
        
        # Product-Customer-Location combination
        if product_id and customer_id and location_id:
            combo_key = f"{product_id}_{customer_id}_{location_id}"
            if combo_key in DimensionManager._combination_cache['product_customer_location']:
                result['product_customer_location_id'] = DimensionManager._combination_cache['product_customer_location'][combo_key]
            else:
                combo_key_hash = generate_combination_key(product_id, customer_id, location_id)
                record = db.query(ProductCustomerLocationCombination).filter(
                    ProductCustomerLocationCombination.combination_key == combo_key_hash
                ).first()
                if not record:
                    record = ProductCustomerLocationCombination(
                        product_id=product_id,
                        customer_id=customer_id,
                        location_id=location_id,
                        combination_key=combo_key_hash
                    )
                    db.add(record)
                    db.flush()
                DimensionManager._combination_cache['product_customer_location'][combo_key] = record.id
                result['product_customer_location_id'] = record.id
        
        return result
    
    @staticmethod
    def process_forecast_row(db: Session, row_data: dict) -> dict:
        """Process a single forecast data row and return surrogate keys"""
        result = {}
        
        # Extract dimension values
        product_name = str(row_data.get('product')) if row_data.get('product') is not None else None
        customer_name = str(row_data.get('customer')) if row_data.get('customer') is not None else None
        location_name = str(row_data.get('location')) if row_data.get('location') is not None else None
        
        # Get or create dimension IDs
        if product_name:
            result['product_id'] = DimensionManager.get_or_create_dimension_cached(
                db, 'product', product_name,
                product_group=row_data.get('product_group'),
                product_hierarchy=row_data.get('product_hierarchy')
            )
        
        if customer_name:
            result['customer_id'] = DimensionManager.get_or_create_dimension_cached(
                db, 'customer', customer_name,
                customer_group=row_data.get('customer_group'),
                customer_region=row_data.get('customer_region'),
                ship_to_party=row_data.get('ship_to_party'),
                sold_to_party=row_data.get('sold_to_party')
            )
        
        if location_name:
            result['location_id'] = DimensionManager.get_or_create_dimension_cached(
                db, 'location', location_name,
                location_region=row_data.get('location_region')
            )
        
        # Get combination IDs
        if result.get('product_id') or result.get('customer_id') or result.get('location_id'):
            combination_ids = DimensionManager.get_combination_ids(
                db,
                product_id=result.get('product_id'),
                customer_id=result.get('customer_id'),
                location_id=result.get('location_id')
            )
            result.update(combination_ids)
        
        return result
    
    @staticmethod
    def get_dimension_name(db: Session, dimension_type: str, dimension_id: int) -> str:
        """Get human-readable name from dimension ID"""
        if dimension_type == 'product':
            record = db.query(ProductDimension).filter(ProductDimension.id == dimension_id).first()
            return record.product_name if record else f"Product_{dimension_id}"
        elif dimension_type == 'customer':
            record = db.query(CustomerDimension).filter(CustomerDimension.id == dimension_id).first()
            return record.customer_name if record else f"Customer_{dimension_id}"
        elif dimension_type == 'location':
            record = db.query(LocationDimension).filter(LocationDimension.id == dimension_id).first()
            return record.location_name if record else f"Location_{dimension_id}"
        return f"Unknown_{dimension_id}"
    
    @staticmethod
    def get_dimension_id(db: Session, dimension_type: str, dimension_name: str) -> Optional[int]:
        """Get dimension ID from human-readable name"""
        if dimension_type == 'product':
            record = db.query(ProductDimension).filter(ProductDimension.product_name == dimension_name).first()
        elif dimension_type == 'customer':
            record = db.query(CustomerDimension).filter(CustomerDimension.customer_name == dimension_name).first()
        elif dimension_type == 'location':
            record = db.query(LocationDimension).filter(LocationDimension.location_name == dimension_name).first()
        else:
            return None
        return record.id if record else None
def get_distinct_products(db: Session) -> list:
    """Get distinct products"""
    return db.query(ProductDimension.product_name).distinct().all()

def get_distinct_customers(db: Session) -> list:
    """Get distinct customers"""
    return db.query(CustomerDimension.customer_name).distinct().all()

def get_distinct_locations(db: Session) -> list:
    """Get distinct locations"""
    return db.query(LocationDimension.location_name).distinct().all()
