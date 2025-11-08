"""
Setup Script for Enhanced Horizon AI POS System
==============================================

This script sets up the complete enhanced POS system with:
- Inventory management
- Sales workflow
- AI model training
- Sample data generation

Author: Horizon Enterprise Team
Course: AI for Software Engineering  
Date: November 8, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_enhanced_system():
    """Setup the complete enhanced POS system"""
    print("🔧 ENHANCED HORIZON POS SYSTEM SETUP")
    print("=" * 50)
    
    try:
        # Create all necessary directories
        directories = ['data', 'models', 'logs', 'reports', 'src/inventory']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        
        # Setup inventory system
        setup_initial_inventory()
        
        # Create sample sales data
        create_sample_sales_data()
        
        # Train AI models with enhanced data
        train_enhanced_models()
        
        print("\n" + "=" * 50)
        print("🎉 ENHANCED SYSTEM SETUP COMPLETED!")
        print("=" * 50)
        print("✅ Inventory system ready")
        print("✅ Sample data created")
        print("✅ AI models trained")
        print("✅ All directories created")
        print("\n🚀 Run 'python enhanced_main.py' to start the system!")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def setup_initial_inventory():
    """Create initial inventory with comprehensive product data"""
    print("\n📦 Setting up initial inventory...")
    
    # Enhanced product catalog
    products = [
        # Electronics
        {'product_id': 'ELEC_001', 'product_name': 'iPhone 15 Pro', 'category': 'Electronics', 
         'unit_price': 1199.99, 'cost_price': 900.00, 'current_stock': 25, 'minimum_stock': 5, 'supplier': 'Apple Inc'},
        {'product_id': 'ELEC_002', 'product_name': 'Samsung Galaxy S24', 'category': 'Electronics', 
         'unit_price': 1099.99, 'cost_price': 800.00, 'current_stock': 30, 'minimum_stock': 5, 'supplier': 'Samsung'},
        {'product_id': 'ELEC_003', 'product_name': 'iPad Air', 'category': 'Electronics', 
         'unit_price': 799.99, 'cost_price': 600.00, 'current_stock': 20, 'minimum_stock': 3, 'supplier': 'Apple Inc'},
        {'product_id': 'ELEC_004', 'product_name': 'AirPods Pro 2', 'category': 'Electronics', 
         'unit_price': 279.99, 'cost_price': 200.00, 'current_stock': 50, 'minimum_stock': 10, 'supplier': 'Apple Inc'},
        {'product_id': 'ELEC_005', 'product_name': 'Sony WH-1000XM5', 'category': 'Electronics', 
         'unit_price': 399.99, 'cost_price': 280.00, 'current_stock': 15, 'minimum_stock': 5, 'supplier': 'Sony'},
        
        # Computers
        {'product_id': 'COMP_001', 'product_name': 'MacBook Air M3', 'category': 'Computers', 
         'unit_price': 1399.99, 'cost_price': 1100.00, 'current_stock': 12, 'minimum_stock': 3, 'supplier': 'Apple Inc'},
        {'product_id': 'COMP_002', 'product_name': 'Dell XPS 13', 'category': 'Computers', 
         'unit_price': 1299.99, 'cost_price': 950.00, 'current_stock': 15, 'minimum_stock': 3, 'supplier': 'Dell'},
        {'product_id': 'COMP_003', 'product_name': 'HP Spectre x360', 'category': 'Computers', 
         'unit_price': 1199.99, 'cost_price': 850.00, 'current_stock': 10, 'minimum_stock': 2, 'supplier': 'HP'},
        {'product_id': 'COMP_004', 'product_name': 'Microsoft Surface Pro 9', 'category': 'Computers', 
         'unit_price': 1099.99, 'cost_price': 800.00, 'current_stock': 18, 'minimum_stock': 4, 'supplier': 'Microsoft'},
        
        # Accessories
        {'product_id': 'ACC_001', 'product_name': 'Magic Mouse', 'category': 'Accessories', 
         'unit_price': 99.99, 'cost_price': 65.00, 'current_stock': 35, 'minimum_stock': 10, 'supplier': 'Apple Inc'},
        {'product_id': 'ACC_002', 'product_name': 'USB-C Hub', 'category': 'Accessories', 
         'unit_price': 49.99, 'cost_price': 25.00, 'current_stock': 40, 'minimum_stock': 15, 'supplier': 'Anker'},
        {'product_id': 'ACC_003', 'product_name': 'Wireless Charger', 'category': 'Accessories', 
         'unit_price': 39.99, 'cost_price': 20.00, 'current_stock': 60, 'minimum_stock': 20, 'supplier': 'Belkin'},
        {'product_id': 'ACC_004', 'product_name': 'Phone Case iPhone 15', 'category': 'Accessories', 
         'unit_price': 24.99, 'cost_price': 12.00, 'current_stock': 100, 'minimum_stock': 25, 'supplier': 'OtterBox'},
        {'product_id': 'ACC_005', 'product_name': 'Screen Protector', 'category': 'Accessories', 
         'unit_price': 19.99, 'cost_price': 8.00, 'current_stock': 80, 'minimum_stock': 30, 'supplier': 'ZAGG'},
        
        # Smart Home
        {'product_id': 'SMART_001', 'product_name': 'Echo Dot (5th Gen)', 'category': 'Smart_Home', 
         'unit_price': 59.99, 'cost_price': 35.00, 'current_stock': 45, 'minimum_stock': 15, 'supplier': 'Amazon'},
        {'product_id': 'SMART_002', 'product_name': 'Google Nest Hub', 'category': 'Smart_Home', 
         'unit_price': 129.99, 'cost_price': 80.00, 'current_stock': 25, 'minimum_stock': 8, 'supplier': 'Google'},
        {'product_id': 'SMART_003', 'product_name': 'Philips Hue Bulbs (4-pack)', 'category': 'Smart_Home', 
         'unit_price': 89.99, 'cost_price': 55.00, 'current_stock': 30, 'minimum_stock': 10, 'supplier': 'Philips'},
        
        # Gaming
        {'product_id': 'GAME_001', 'product_name': 'PlayStation 5', 'category': 'Gaming', 
         'unit_price': 499.99, 'cost_price': 400.00, 'current_stock': 8, 'minimum_stock': 2, 'supplier': 'Sony'},
        {'product_id': 'GAME_002', 'product_name': 'Xbox Series X', 'category': 'Gaming', 
         'unit_price': 499.99, 'cost_price': 400.00, 'current_stock': 6, 'minimum_stock': 2, 'supplier': 'Microsoft'},
        {'product_id': 'GAME_003', 'product_name': 'Nintendo Switch OLED', 'category': 'Gaming', 
         'unit_price': 349.99, 'cost_price': 250.00, 'current_stock': 12, 'minimum_stock': 3, 'supplier': 'Nintendo'}
    ]
    
    # Add timestamp
    for product in products:
        product['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create DataFrame and save
    inventory_df = pd.DataFrame(products)
    inventory_df.to_csv('data/inventory.csv', index=False)
    
    print(f"✅ Created inventory with {len(products)} products")
    return True

def create_sample_sales_data():
    """Create comprehensive sample sales data"""
    print("\n💰 Creating sample sales data...")
    
    # Load inventory
    inventory = pd.read_csv('data/inventory.csv')
    
    # Generate sales data for the last 30 days
    sales_data = []
    base_date = datetime.now() - timedelta(days=30)
    
    sales_assistants = ['Alice Johnson', 'Bob Smith', 'Carol Brown', 'David Wilson', 'Eva Davis']
    payment_methods = ['Cash', 'Card', 'Digital']
    
    for day in range(30):
        current_date = base_date + timedelta(days=day)
        
        # More sales on weekends
        daily_transactions = np.random.poisson(8 if current_date.weekday() < 5 else 15)
        
        for transaction in range(daily_transactions):
            # Select random product (weighted by stock levels)
            available_products = inventory[inventory['current_stock'] > 0]
            if available_products.empty:
                continue
            
            weights = available_products['current_stock'] / available_products['current_stock'].sum()
            selected_product = available_products.sample(weights=weights).iloc[0]
            
            # Generate transaction details
            quantity = np.random.randint(1, min(4, selected_product['current_stock'] + 1))
            unit_price = selected_product['unit_price']
            
            # Apply random discount (10% chance)
            discount_percent = np.random.choice([0, 5, 10, 15], p=[0.7, 0.15, 0.1, 0.05])
            subtotal = quantity * unit_price
            discount_amount = subtotal * (discount_percent / 100)
            total_amount = subtotal - discount_amount
            
            # Transaction time (business hours)
            transaction_time = current_date.replace(
                hour=np.random.randint(9, 19),
                minute=np.random.randint(0, 60),
                second=np.random.randint(0, 60)
            )
            
            sale_record = {
                'transaction_id': f"TXN_{transaction_time.strftime('%Y%m%d_%H%M%S')}_{transaction:03d}",
                'date': current_date.strftime('%Y-%m-%d'),
                'time': transaction_time.strftime('%H:%M:%S'),
                'customer_id': f"CUST_{np.random.randint(1000, 9999)}",
                'product_id': selected_product['product_id'],
                'product_name': selected_product['product_name'],
                'category': selected_product['category'],
                'quantity': quantity,
                'unit_price': unit_price,
                'total_amount': total_amount,
                'payment_method': np.random.choice(payment_methods),
                'sales_assistant': np.random.choice(sales_assistants),
                'discount_applied': discount_percent
            }
            
            sales_data.append(sale_record)
    
    # Create DataFrame and save
    sales_df = pd.DataFrame(sales_data)
    sales_df.to_csv('data/daily_sales.csv', index=False)
    
    print(f"✅ Created {len(sales_data)} sample sales transactions")
    return True

def train_enhanced_models():
    """Train AI models with enhanced data"""
    print("\n🤖 Training AI models with enhanced data...")
    
    try:
        # Add src to path
        sys.path.append('src')
        
        # Load enhanced transaction data
        sales_data = pd.read_csv('data/daily_sales.csv')
        inventory_data = pd.read_csv('data/inventory.csv')
        
        # Merge for complete dataset
        enhanced_data = sales_data.merge(
            inventory_data[['product_id', 'cost_price', 'supplier']], 
            on='product_id', 
            how='left'
        )
        
        # Add timestamp columns
        enhanced_data['transaction_timestamp'] = pd.to_datetime(
            enhanced_data['date'] + ' ' + enhanced_data['time']
        )
        enhanced_data['transaction_date'] = pd.to_datetime(enhanced_data['date'])
        
        # Train Sales Predictor
        print("📈 Training Sales Predictor...")
        from ai_models.sales_predictor import SalesPredictor
        sales_predictor = SalesPredictor()
        sales_predictor.train(data=enhanced_data)
        sales_predictor.save_model('models/sales_predictor.pkl')
        print("✅ Sales Predictor trained")
        
        # Train Customer Segmentation
        print("👥 Training Customer Segmentation...")
        from ai_models.customer_segmentation import CustomerSegmentation
        customer_segmentation = CustomerSegmentation(n_clusters=4)
        customer_segmentation.train(data=enhanced_data)
        customer_segmentation.save_model('models/customer_segmentation.pkl')
        print("✅ Customer Segmentation trained")
        
        # Train Fraud Detector
        print("🛡️ Training Fraud Detector...")
        from ai_models.fraud_detector_fixed import FraudDetectorFixed
        fraud_detector = FraudDetectorFixed()
        fraud_detector.train(data=enhanced_data)
        fraud_detector.save_model('models/fraud_detector.pkl')
        print("✅ Fraud Detector trained")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_user_guide():
    """Create a user guide for the enhanced system"""
    guide_content = """
# Horizon AI POS System - User Guide

## System Overview
The Enhanced Horizon AI POS System provides comprehensive business management with:
- Inventory management and stock tracking
- Sales processing with AI insights
- Manager dashboard with analytics
- Staff performance tracking

## Getting Started

### For Managers:
1. Run `python enhanced_main.py`
2. Select "Manager Dashboard" from main menu
3. Access inventory management, reports, and analytics

### For Sales Assistants:
1. Run `python enhanced_main.py`
2. Select "Sales Assistant Mode"
3. Enter your name and start processing sales

## Key Features

### Inventory Management:
- Add new products with full details
- Update stock levels in real-time
- Monitor low stock alerts
- Generate inventory reports

### Sales Processing:
- Select products from available inventory
- Automatic stock updates after each sale
- Apply discounts and process payments
- AI-powered fraud detection

### Analytics & Reports:
- Daily sales summaries
- Staff performance metrics
- Revenue analysis by category
- Inventory valuation reports

## Data Files:
- `data/inventory.csv` - Product inventory
- `data/daily_sales.csv` - Sales transactions
- `models/` - Trained AI models
- `reports/` - Generated reports

## Support:
For technical support, check the logs in the `logs/` directory.
"""
    
    with open('USER_GUIDE.md', 'w') as f:
        f.write(guide_content)
    
    print("✅ User guide created: USER_GUIDE.md")

def main():
    """Main setup function"""
    print("Starting Enhanced Horizon POS System Setup...")
    
    if setup_enhanced_system():
        create_user_guide()
        print("\n🎉 Setup completed successfully!")
        print("📖 Read USER_GUIDE.md for usage instructions")
        print("🚀 Run 'python enhanced_main.py' to start the system")
        return True
    else:
        print("\n❌ Setup failed!")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)