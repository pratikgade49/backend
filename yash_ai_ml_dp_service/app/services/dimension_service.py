#!/usr/bin/env python3
"""
Dimension service - handles product, customer, location dimensions
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.repositories.dimension_repository import DimensionManager
from app.models.dimension import ProductDimension, CustomerDimension, LocationDimension

class DimensionService:
    """Service for managing dimensions"""

    @staticmethod
    def get_all_products(db: Session) -> List[str]:
        """Get all unique product names"""
        products = db.query(ProductDimension.product_name).distinct().all()
        return [p[0] for p in products]

    @staticmethod
    def get_all_customers(db: Session) -> List[str]:
        """Get all unique customer names"""
        customers = db.query(CustomerDimension.customer_name).distinct().all()
        return [c[0] for c in customers]

    @staticmethod
    def get_all_locations(db: Session) -> List[str]:
        """Get all unique location names"""
        locations = db.query(LocationDimension.location_name).distinct().all()
        return [l[0] for l in locations]

    @staticmethod
    def get_product_id(db: Session, product_name: str) -> Optional[int]:
        """Get product ID by name"""
        return DimensionManager.get_dimension_id(db, 'product', product_name)

    @staticmethod
    def get_customer_id(db: Session, customer_name: str) -> Optional[int]:
        """Get customer ID by name"""
        return DimensionManager.get_dimension_id(db, 'customer', customer_name)

    @staticmethod
    def get_location_id(db: Session, location_name: str) -> Optional[int]:
        """Get location ID by name"""
        return DimensionManager.get_dimension_id(db, 'location', location_name)

    @staticmethod
    def create_product(db: Session, product_name: str) -> ProductDimension:
        """Create a new product dimension"""
        product = ProductDimension(product_name=product_name)
        db.add(product)
        db.commit()
        db.refresh(product)
        return product

    @staticmethod
    def create_customer(db: Session, customer_name: str) -> CustomerDimension:
        """Create a new customer dimension"""
        customer = CustomerDimension(customer_name=customer_name)
        db.add(customer)
        db.commit()
        db.refresh(customer)
        return customer

    @staticmethod
    def create_location(db: Session, location_name: str) -> LocationDimension:
        """Create a new location dimension"""
        location = LocationDimension(location_name=location_name)
        db.add(location)
        db.commit()
        db.refresh(location)
        return location

    @staticmethod
    def get_or_create_product(db: Session, product_name: str) -> int:
        """Get or create product and return ID"""
        return DimensionManager.get_or_create_dimension(db, 'product', product_name)

    @staticmethod
    def get_or_create_customer(db: Session, customer_name: str) -> int:
        """Get or create customer and return ID"""
        return DimensionManager.get_or_create_dimension(db, 'customer', customer_name)

    @staticmethod
    def get_or_create_location(db: Session, location_name: str) -> int:
        """Get or create location and return ID"""
        return DimensionManager.get_or_create_dimension(db, 'location', location_name)
