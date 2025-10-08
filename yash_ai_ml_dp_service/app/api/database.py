#!/usr/bin/env python3
"""
Database statistics and data viewing API routes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any, Optional

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.forecast import ForecastData
from app.models.dimension import ProductDimension, CustomerDimension, LocationDimension
from app.schemas.stats import DataViewRequest, DataViewResponse, DatabaseStats

router = APIRouter(prefix="/database", tags=["Database"])

@router.get("/stats", response_model=DatabaseStats)
async def get_database_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get database statistics"""
    total_records = db.query(func.count(ForecastData.id)).scalar()
    total_products = db.query(func.count(ProductDimension.id)).scalar()
    total_customers = db.query(func.count(CustomerDimension.id)).scalar()
    total_locations = db.query(func.count(LocationDimension.id)).scalar()
    total_users = db.query(func.count(User.id)).scalar()
    
    # Get date range
    date_range = db.query(
        func.min(ForecastData.date).label('min_date'),
        func.max(ForecastData.date).label('max_date')
    ).first()
    
    return DatabaseStats(
        total_records=total_records or 0,
        total_products=total_products or 0,
        total_customers=total_customers or 0,
        total_locations=total_locations or 0,
        total_users=total_users or 0,
        date_range={
            "start": str(date_range.min_date) if date_range.min_date else None,
            "end": str(date_range.max_date) if date_range.max_date else None
        },
        total_saved_forecasts=0  # Add this when saved_forecasts model is ready
    )

@router.get("/options")
async def get_database_options(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all available filter options"""
    products = db.query(ProductDimension.name).distinct().all()
    customers = db.query(CustomerDimension.name).distinct().all()
    locations = db.query(LocationDimension.name).distinct().all()
    
    return {
        "products": [p.name for p in products],
        "customers": [c.name for c in customers],
        "locations": [l.name for l in locations]
    }

@router.post("/filtered_options")
async def get_filtered_options(
    filters: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get filtered options based on current selections"""
    query = db.query(ForecastData)
    
    # Apply filters
    if filters.get('product'):
        product = db.query(ProductDimension).filter(
            ProductDimension.name == filters['product']
        ).first()
        if product:
            query = query.filter(ForecastData.product_id == product.id)
    
    if filters.get('customer'):
        customer = db.query(CustomerDimension).filter(
            CustomerDimension.name == filters['customer']
        ).first()
        if customer:
            query = query.filter(ForecastData.customer_id == customer.id)
    
    if filters.get('location'):
        location = db.query(LocationDimension).filter(
            LocationDimension.name == filters['location']
        ).first()
        if location:
            query = query.filter(ForecastData.location_id == location.id)
    
    # Get available options for remaining filters
    available_products = set()
    available_customers = set()
    available_locations = set()
    
    for record in query.all():
        if record.product_id:
            product = db.query(ProductDimension).get(record.product_id)
            if product:
                available_products.add(product.name)
        if record.customer_id:
            customer = db.query(CustomerDimension).get(record.customer_id)
            if customer:
                available_customers.add(customer.name)
        if record.location_id:
            location = db.query(LocationDimension).get(record.location_id)
            if location:
                available_locations.add(location.name)
    
    return {
        "products": sorted(list(available_products)),
        "customers": sorted(list(available_customers)),
        "locations": sorted(list(available_locations))
    }

@router.post("/view", response_model=DataViewResponse)
async def view_data(
    request: DataViewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """View filtered data from database"""
    query = db.query(ForecastData)
    
    # Apply filters
    if request.product:
        product = db.query(ProductDimension).filter(
            ProductDimension.name == request.product
        ).first()
        if product:
            query = query.filter(ForecastData.product_id == product.id)
    
    if request.customer:
        customer = db.query(CustomerDimension).filter(
            CustomerDimension.name == request.customer
        ).first()
        if customer:
            query = query.filter(ForecastData.customer_id == customer.id)
    
    if request.location:
        location = db.query(LocationDimension).filter(
            LocationDimension.name == request.location
        ).first()
        if location:
            query = query.filter(ForecastData.location_id == location.id)
    
    if request.start_date:
        query = query.filter(ForecastData.date >= request.start_date)
    
    if request.end_date:
        query = query.filter(ForecastData.date <= request.end_date)
    
    total_count = query.count()
    
    # Apply limit
    records = query.order_by(ForecastData.date.desc()).limit(request.limit).all()
    
    # Format data
    data = []
    for record in records:
        product_name = None
        customer_name = None
        location_name = None
        
        if record.product_id:
            product = db.query(ProductDimension).get(record.product_id)
            product_name = product.name if product else None
        
        if record.customer_id:
            customer = db.query(CustomerDimension).get(record.customer_id)
            customer_name = customer.name if customer else None
        
        if record.location_id:
            location = db.query(LocationDimension).get(record.location_id)
            location_name = location.name if location else None
        
        data.append({
            "id": record.id,
            "date": str(record.date),
            "quantity": record.quantity,
            "product": product_name,
            "customer": customer_name,
            "location": location_name
        })
    
    return DataViewResponse(
        data=data,
        total_count=total_count,
        filtered_count=len(data)
    )
