#!/usr/bin/env python3
"""
Dimension tables for surrogate key system
"""

from sqlalchemy import Column, Integer, String, DateTime, UniqueConstraint
from datetime import datetime
from app.core.database import Base

class ProductDimension(Base):
    """Dimension table for products with surrogate keys"""
    __tablename__ = "product_dimension"
    
    id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), unique=True, nullable=False, index=True)
    product_group = Column(String(255), nullable=True)
    product_hierarchy = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CustomerDimension(Base):
    """Dimension table for customers with surrogate keys"""
    __tablename__ = "customer_dimension"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_name = Column(String(255), unique=True, nullable=False, index=True)
    customer_group = Column(String(255), nullable=True)
    customer_region = Column(String(255), nullable=True)
    ship_to_party = Column(String(255), nullable=True)
    sold_to_party = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LocationDimension(Base):
    """Dimension table for locations with surrogate keys"""
    __tablename__ = "location_dimension"
    
    id = Column(Integer, primary_key=True, index=True)
    location_name = Column(String(255), unique=True, nullable=False, index=True)
    location_region = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ProductCustomerCombination(Base):
    """2D combination table for product-customer pairs"""
    __tablename__ = "product_customer_combination"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False, index=True)
    customer_id = Column(Integer, nullable=False, index=True)
    combination_key = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('product_id', 'customer_id', name='unique_product_customer'),
    )

class ProductLocationCombination(Base):
    """2D combination table for product-location pairs"""
    __tablename__ = "product_location_combination"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False, index=True)
    location_id = Column(Integer, nullable=False, index=True)
    combination_key = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('product_id', 'location_id', name='unique_product_location'),
    )

class CustomerLocationCombination(Base):
    """2D combination table for customer-location pairs"""
    __tablename__ = "customer_location_combination"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, nullable=False, index=True)
    location_id = Column(Integer, nullable=False, index=True)
    combination_key = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('customer_id', 'location_id', name='unique_customer_location'),
    )

class ProductCustomerLocationCombination(Base):
    """3D combination table for product-customer-location triplets"""
    __tablename__ = "product_customer_location_combination"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False, index=True)
    customer_id = Column(Integer, nullable=False, index=True)
    location_id = Column(Integer, nullable=False, index=True)
    combination_key = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('product_id', 'customer_id', 'location_id', name='unique_product_customer_location'),
    )
