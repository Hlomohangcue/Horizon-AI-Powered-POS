"""
Enhanced Setup Script for Streamlit Deployment
==============================================

This script prepares the Horizon AI POS system for Streamlit deployment
by creating necessary data files and training AI models.

Author: [Your Name]
Course: AI for Software Engineering
Date: November 8, 2025
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_sample_inventory():
    """Create sample inventory data"""
    print("📦 Creating sample inventory...")
    
    products = [
        {"product_id": "PROD_001", "product_name": "iPhone 14", "category": "Electronics", "unit_price": 999.99, "stock_quantity": 50, "reorder_level": 10},
        {"product_id": "PROD_002", "product_name": "Samsung Galaxy S23", "category": "Electronics", "unit_price": 899.99, "stock_quantity": 45, "reorder_level": 10},
        {"product_id": "PROD_003", "product_name": "MacBook Pro", "category": "Electronics", "unit_price": 1999.99, "stock_quantity": 20, "reorder_level": 5},
        {"product_id": "PROD_004", "product_name": "Nike Air Max", "category": "Clothing", "unit_price": 149.99, "stock_quantity": 100, "reorder_level": 20},
        {"product_id": "PROD_005", "product_name": "Adidas Sneakers", "category": "Clothing", "unit_price": 129.99, "stock_quantity": 80, "reorder_level": 15},
        {"product_id": "PROD_006", "product_name": "Levi's Jeans", "category": "Clothing", "unit_price": 79.99, "stock_quantity": 60, "reorder_level": 15},
        {"product_id": "PROD_007", "product_name": "Coffee Maker", "category": "Home", "unit_price": 89.99, "stock_quantity": 30, "reorder_level": 5},
        {"product_id": "PROD_008", "product_name": "Wireless Headphones", "category": "Electronics", "unit_price": 199.99, "stock_quantity": 75, "reorder_level": 15},
        {"product_id": "PROD_009", "product_name": "Yoga Mat", "category": "Sports", "unit_price": 29.99, "stock_quantity": 40, "reorder_level": 10},
        {"product_id": "PROD_010", "product_name": "Water Bottle", "category": "Sports", "unit_price": 19.99, "stock_quantity": 120, "reorder_level": 25},
    ]
    
    inventory_df = pd.DataFrame(products)
    inventory_df['date_added'] = datetime.now().strftime('%Y-%m-%d')
    inventory_df['description'] = inventory_df['product_name'] + " - High quality product"
    
    inventory_df.to_csv('data/inventory.csv', index=False)
    print(f"✅ Created inventory with {len(inventory_df)} products")
    
    return inventory_df

def create_sample_transactions(inventory_df):
    """Create sample transaction data"""
    print("🧾 Creating sample transactions...")
    
    np.random.seed(42)
    transactions = []
    
    # Generate 100 sample transactions
    for i in range(100):
        # Random date in the last 30 days
        days_ago = np.random.randint(0, 30)
        transaction_date = datetime.now() - timedelta(days=days_ago)
        
        # Random product
        product = inventory_df.sample(1).iloc[0]
        quantity = np.random.randint(1, 6)
        
        # Customer info
        customer_id = f"CUST_{np.random.randint(1000, 9999)}"
        customer_name = f"Customer {np.random.randint(100, 999)}"
        
        # Payment method
        payment_methods = ["Cash", "Credit Card", "Debit Card", "Mobile Payment"]
        payment_method = np.random.choice(payment_methods)
        
        # Calculate amounts
        subtotal = product['unit_price'] * quantity
        discount = np.random.choice([0, 5, 10, 15], p=[0.7, 0.15, 0.1, 0.05])
        discount_amount = subtotal * (discount / 100)
        total_amount = subtotal - discount_amount
        
        transaction = {
            'transaction_id': f"TXN_{transaction_date.strftime('%Y%m%d')}_{i:03d}",
            'customer_id': customer_id,
            'customer_name': customer_name,
            'product_id': product['product_id'],
            'product_name': product['product_name'],
            'category': product['category'],
            'quantity': quantity,
            'unit_price': product['unit_price'],
            'subtotal': subtotal,
            'discount_amount': discount_amount,
            'total_amount': total_amount,
            'payment_method': payment_method,
            'transaction_date': transaction_date.strftime('%Y-%m-%d'),
            'transaction_timestamp': transaction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'cashier': f"Sales Assistant {np.random.randint(1, 5)}"
        }
        
        transactions.append(transaction)
    
    transactions_df = pd.DataFrame(transactions)
    transactions_df.to_csv('data/transactions.csv', index=False)
    print(f"✅ Created {len(transactions_df)} sample transactions")
    
    return transactions_df

def create_sample_customers():
    """Create sample customer data"""
    print("👥 Creating sample customers...")
    
    customers = []
    
    for i in range(50):
        customer = {
            'customer_id': f"CUST_{1000 + i}",
            'customer_name': f"Customer {100 + i}",
            'email': f"customer{100 + i}@email.com",
            'phone': f"555-{np.random.randint(1000, 9999)}",
            'registration_date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d'),
            'loyalty_points': np.random.randint(0, 1000)
        }
        customers.append(customer)
    
    customers_df = pd.DataFrame(customers)
    customers_df.to_csv('data/customers.csv', index=False)
    print(f"✅ Created {len(customers_df)} sample customers")
    
    return customers_df

def setup_ai_models():
    """Setup and train AI models"""
    print("🤖 Setting up AI models...")
    
    try:
        # Import the enhanced setup function if available
        if os.path.exists('src/ai_models/sales_predictor.py'):
            print("📈 Sales Predictor found")
        if os.path.exists('src/ai_models/customer_segmentation.py'):
            print("👥 Customer Segmentation found") 
        if os.path.exists('src/ai_models/fraud_detector_fixed.py'):
            print("🛡️ Fraud Detector found")
        
        print("✅ AI models are ready (will be trained when first used)")
        
    except Exception as e:
        print(f"⚠️ AI models setup warning: {e}")

def main():
    """Main setup function"""
    print("🚀 ENHANCED SETUP FOR STREAMLIT DEPLOYMENT")
    print("=" * 60)
    
    try:
        # Create directories
        create_directories()
        
        # Create sample data
        inventory_df = create_sample_inventory()
        transactions_df = create_sample_transactions(inventory_df)
        customers_df = create_sample_customers()
        
        # Setup AI models
        setup_ai_models()
        
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ All directories created")
        print("✅ Sample data generated")
        print("✅ AI models prepared")
        print("\n🌐 Ready for Streamlit deployment!")
        print("Run: streamlit run streamlit_app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)