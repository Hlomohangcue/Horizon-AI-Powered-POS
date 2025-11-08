"""
Inventory Management System for Horizon AI POS
=============================================

This module handles all inventory operations including:
- Stock tracking and management
- Product addition and updates
- Low stock alerts
- Inventory reports

Author: Horizon Enterprise Team
Course: AI for Software Engineering
Date: November 8, 2025
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class InventoryManager:
    """
    Comprehensive inventory management system for Horizon POS
    
    Handles stock levels, product management, and inventory analytics
    """
    
    def __init__(self, inventory_file='data/inventory.csv'):
        """Initialize inventory manager"""
        self.inventory_file = inventory_file
        self.ensure_inventory_file()
        logger.info("Inventory Manager initialized")
    
    def ensure_inventory_file(self):
        """Create inventory file if it doesn't exist"""
        if not os.path.exists(self.inventory_file):
            # Create sample inventory
            initial_inventory = pd.DataFrame({
                'product_id': ['PROD_001', 'PROD_002', 'PROD_003', 'PROD_004', 'PROD_005'],
                'product_name': ['iPhone 14', 'Samsung Galaxy S23', 'MacBook Air', 'Dell Laptop', 'AirPods Pro'],
                'category': ['Electronics', 'Electronics', 'Computers', 'Computers', 'Accessories'],
                'unit_price': [999.99, 899.99, 1299.99, 799.99, 249.99],
                'current_stock': [25, 30, 15, 20, 50],
                'minimum_stock': [5, 5, 3, 5, 10],
                'supplier': ['Apple Inc', 'Samsung', 'Apple Inc', 'Dell', 'Apple Inc'],
                'last_updated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 5,
                'cost_price': [800.00, 650.00, 1000.00, 600.00, 180.00]
            })
            
            os.makedirs(os.path.dirname(self.inventory_file), exist_ok=True)
            initial_inventory.to_csv(self.inventory_file, index=False)
            logger.info("Created initial inventory file")
    
    def load_inventory(self):
        """Load current inventory from CSV"""
        try:
            return pd.read_csv(self.inventory_file)
        except Exception as e:
            logger.error(f"Error loading inventory: {e}")
            return pd.DataFrame()
    
    def save_inventory(self, inventory_df):
        """Save inventory to CSV"""
        try:
            inventory_df['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            inventory_df.to_csv(self.inventory_file, index=False)
            logger.info("Inventory saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving inventory: {e}")
            return False
    
    def add_new_product(self, product_data):
        """
        Add new product to inventory
        
        Args:
            product_data (dict): Product information
            
        Returns:
            bool: Success status
        """
        try:
            inventory = self.load_inventory()
            
            # Generate product ID if not provided
            if 'product_id' not in product_data or not product_data['product_id']:
                max_id = inventory['product_id'].str.extract('(\d+)').astype(int).max().iloc[0] if not inventory.empty else 0
                product_data['product_id'] = f"PROD_{max_id + 1:03d}"
            
            # Check if product already exists
            if product_data['product_id'] in inventory['product_id'].values:
                logger.warning(f"Product {product_data['product_id']} already exists")
                return False
            
            # Add new product
            new_product = pd.DataFrame([product_data])
            inventory = pd.concat([inventory, new_product], ignore_index=True)
            
            return self.save_inventory(inventory)
            
        except Exception as e:
            logger.error(f"Error adding new product: {e}")
            return False
    
    def update_stock(self, product_id, quantity_change, operation='subtract'):
        """
        Update stock levels for a product
        
        Args:
            product_id (str): Product identifier
            quantity_change (int): Quantity to add/subtract
            operation (str): 'add' or 'subtract'
            
        Returns:
            bool: Success status
        """
        try:
            inventory = self.load_inventory()
            
            if product_id not in inventory['product_id'].values:
                logger.error(f"Product {product_id} not found")
                return False
            
            # Update stock
            idx = inventory[inventory['product_id'] == product_id].index[0]
            current_stock = inventory.loc[idx, 'current_stock']
            
            if operation == 'add':
                new_stock = current_stock + quantity_change
            else:  # subtract
                new_stock = max(0, current_stock - quantity_change)
            
            inventory.loc[idx, 'current_stock'] = new_stock
            
            return self.save_inventory(inventory)
            
        except Exception as e:
            logger.error(f"Error updating stock: {e}")
            return False
    
    def get_available_products(self, in_stock_only=True):
        """
        Get list of available products
        
        Args:
            in_stock_only (bool): Only return products with stock > 0
            
        Returns:
            pd.DataFrame: Available products
        """
        inventory = self.load_inventory()
        
        if in_stock_only:
            return inventory[inventory['current_stock'] > 0].copy()
        
        return inventory.copy()
    
    def get_low_stock_alerts(self):
        """Get products with low stock levels"""
        inventory = self.load_inventory()
        
        low_stock = inventory[
            inventory['current_stock'] <= inventory['minimum_stock']
        ].copy()
        
        return low_stock
    
    def get_inventory_value(self):
        """Calculate total inventory value"""
        inventory = self.load_inventory()
        
        if inventory.empty:
            return 0
        
        inventory['total_value'] = inventory['current_stock'] * inventory['cost_price']
        return inventory['total_value'].sum()
    
    def get_product_by_id(self, product_id):
        """Get specific product information"""
        inventory = self.load_inventory()
        
        product = inventory[inventory['product_id'] == product_id]
        
        if product.empty:
            return None
        
        return product.iloc[0].to_dict()
    
    def search_products(self, search_term):
        """Search products by name or category"""
        inventory = self.load_inventory()
        
        if inventory.empty:
            return pd.DataFrame()
        
        # Search in product name and category
        mask = (
            inventory['product_name'].str.contains(search_term, case=False, na=False) |
            inventory['category'].str.contains(search_term, case=False, na=False)
        )
        
        return inventory[mask].copy()
    
    def generate_inventory_report(self):
        """Generate comprehensive inventory report"""
        inventory = self.load_inventory()
        
        if inventory.empty:
            return "No inventory data available"
        
        report = []
        report.append("=" * 60)
        report.append("HORIZON ENTERPRISE - INVENTORY REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        total_products = len(inventory)
        total_value = self.get_inventory_value()
        low_stock_count = len(self.get_low_stock_alerts())
        
        report.append("INVENTORY SUMMARY:")
        report.append(f"• Total Products: {total_products}")
        report.append(f"• Total Inventory Value: ${total_value:,.2f}")
        report.append(f"• Low Stock Alerts: {low_stock_count}")
        report.append("")
        
        # Category breakdown
        report.append("CATEGORY BREAKDOWN:")
        category_stats = inventory.groupby('category').agg({
            'current_stock': 'sum',
            'product_id': 'count'
        }).rename(columns={'product_id': 'product_count'})
        
        for category, stats in category_stats.iterrows():
            report.append(f"• {category}: {stats['product_count']} products, {stats['current_stock']} units")
        
        report.append("")
        
        # Low stock alerts
        low_stock = self.get_low_stock_alerts()
        if not low_stock.empty:
            report.append("LOW STOCK ALERTS:")
            for _, product in low_stock.iterrows():
                report.append(f"• {product['product_name']} ({product['product_id']}): {product['current_stock']} units (min: {product['minimum_stock']})")
        
        return "\n".join(report)
    
    def update_product_info(self, product_id, updates):
        """Update product information"""
        try:
            inventory = self.load_inventory()
            
            if product_id not in inventory['product_id'].values:
                return False
            
            idx = inventory[inventory['product_id'] == product_id].index[0]
            
            for field, value in updates.items():
                if field in inventory.columns:
                    inventory.loc[idx, field] = value
            
            return self.save_inventory(inventory)
            
        except Exception as e:
            logger.error(f"Error updating product info: {e}")
            return False