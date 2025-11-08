"""
Test Script with Real Handwritten Data
=====================================

This script tests the AI POS system using the real handwritten transaction data
from your store records.

Author: [Your Name]
Course: AI for Software Engineering
Date: November 8, 2025
"""

import sys
import os
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_real_data():
    """Test the system with real handwritten data"""
    print("🏪 Testing Horizon AI POS with Real Handwritten Data")
    print("=" * 60)
    
    try:
        # Load real transaction data
        transactions_file = 'data/handwritten_transactions.csv'
        products_file = 'data/handwritten_products.csv'
        
        if os.path.exists(transactions_file):
            transactions = pd.read_csv(transactions_file)
            print(f"✅ Loaded {len(transactions)} real transactions")
            
            # Display summary
            print(f"\n📊 Transaction Summary:")
            print(f"   Date Range: {transactions['transaction_date'].min()} to {transactions['transaction_date'].max()}")
            print(f"   Total Sales: ${transactions['total_amount'].sum():.2f}")
            print(f"   Unique Customers: {transactions['customer_id'].nunique()}")
            
            # Top selling products
            top_products = transactions.groupby('product_name')['quantity'].sum().sort_values(ascending=False).head(5)
            print(f"\n🏆 Top Selling Products:")
            for product, qty in top_products.items():
                print(f"   • {product}: {qty} units")
            
            # Daily sales
            daily_sales = transactions.groupby('transaction_date')['total_amount'].sum()
            print(f"\n📅 Daily Sales:")
            for date, sales in daily_sales.items():
                print(f"   • {date}: ${sales:.2f}")
                
        if os.path.exists(products_file):
            products = pd.read_csv(products_file)
            print(f"\n📦 Product Catalog: {len(products)} products available")
            
            # Category breakdown
            category_counts = products['category'].value_counts()
            print(f"\n📂 Product Categories:")
            for category, count in category_counts.items():
                print(f"   • {category}: {count} products")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        return False

def run_pos_with_real_data():
    """Run the POS system with real data loaded"""
    print("\n🚀 Starting POS System with Real Data...")
    
    try:
        from src.pos_system.pos_interface import POSInterface
        from src.ai_models.sales_predictor import SalesPredictor
        from src.ai_models.customer_segmentation import CustomerSegmentation
        from src.ai_models.fraud_detector import FraudDetector
        
        # Initialize AI models (they'll use synthetic data for training for now)
        print("🤖 Initializing AI models...")
        sales_predictor = SalesPredictor()
        customer_segmentation = CustomerSegmentation()
        fraud_detector = FraudDetector()
        
        # Initialize POS interface
        pos_interface = POSInterface(
            sales_predictor=sales_predictor,
            customer_segmentation=customer_segmentation,
            fraud_detector=fraud_detector
        )
        
        # Load real product data into the POS system
        products_file = 'data/handwritten_products.csv'
        if os.path.exists(products_file):
            products_df = pd.read_csv(products_file)
            
            # Convert to POS inventory format
            pos_interface.inventory = {}
            for _, row in products_df.iterrows():
                pos_interface.inventory[row['product_id']] = {
                    'name': row['product_name'],
                    'category': row['category'],
                    'price': row['unit_price'],
                    'stock': row['stock_quantity']
                }
            
            print(f"✅ Loaded {len(pos_interface.inventory)} real products into POS system")
        
        print("\n🎉 System ready with your real data!")
        print("=" * 50)
        print("Your POS system now includes:")
        print("• Real product catalog from your handwritten records")
        print("• Historical transaction data for analysis")
        print("• AI models ready for intelligent insights")
        print("\nThe system is ready to process new transactions!")
        
        return pos_interface
        
    except Exception as e:
        print(f"❌ Error initializing POS system: {e}")
        return None

if __name__ == "__main__":
    # Test real data loading
    if test_real_data():
        print("\n" + "=" * 60)
        
        # Ask user if they want to start the POS system
        choice = input("\nWould you like to start the POS system? (y/n): ").strip().lower()
        
        if choice == 'y':
            pos_system = run_pos_with_real_data()
            if pos_system:
                print("\n💡 To start the interactive POS interface, run: python main.py")
            else:
                print("❌ Failed to initialize POS system")
        else:
            print("👋 Thank you for testing the real data integration!")
    
    else:
        print("❌ Failed to load real data")
        
    input("\nPress Enter to exit...")