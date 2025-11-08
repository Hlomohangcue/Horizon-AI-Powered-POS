"""
Quick Fix Script for Real Data Training
======================================

This script properly formats your real data for the AI models and trains them.

Author: [Your Name]
Course: AI for Software Engineering
Date: November 8, 2025
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_properly_formatted_data():
    """Create properly formatted training data"""
    print("📊 Formatting real data for AI training...")
    
    try:
        # Load real data
        transactions = pd.read_csv('data/handwritten_transactions.csv')
        
        # Create expanded transaction data with proper column names
        expanded_data = []
        
        for _, row in transactions.iterrows():
            # Create multiple entries to simulate daily data
            base_date = pd.to_datetime(row['transaction_date'])
            
            for i in range(5):  # Create 5 variations
                new_row = {
                    'transaction_id': f"{row['transaction_id']}_{i}",
                    'customer_id': row['customer_id'],
                    'product_id': f"PROD_{hash(row['product_name']) % 1000:03d}",
                    'product_name': row['product_name'],
                    'product_category': row['category'],
                    'quantity': row['quantity'] + np.random.randint(-1, 2),
                    'unit_price': row['unit_price'],
                    'total_amount': row['total_amount'] * (1 + np.random.uniform(-0.1, 0.1)),
                    'payment_method': row['payment_method'],
                    'transaction_timestamp': base_date + timedelta(days=i*7, hours=np.random.randint(9, 18)),
                    'transaction_date': (base_date + timedelta(days=i*7)).date(),
                    'store_location': 'Store_A'
                }
                expanded_data.append(new_row)
        
        expanded_df = pd.DataFrame(expanded_data)
        expanded_df.to_csv('data/formatted_transactions.csv', index=False)
        
        print(f"✅ Created {len(expanded_df)} formatted transactions")
        return True
        
    except Exception as e:
        print(f"❌ Error formatting data: {e}")
        return False

def train_models_simple():
    """Train models with simplified approach"""
    print("🤖 Training AI models (simplified approach)...")
    
    try:
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Load formatted data
        data = pd.read_csv('data/formatted_transactions.csv')
        
        # Train Sales Predictor with basic data
        print("📈 Training Sales Predictor...")
        from src.ai_models.sales_predictor import SalesPredictor
        
        sales_predictor = SalesPredictor()
        sales_predictor.train(data=data)
        sales_predictor.save_model('models/sales_predictor.pkl')
        print("✅ Sales Predictor trained")
        
        # Train Customer Segmentation
        print("👥 Training Customer Segmentation...")
        from src.ai_models.customer_segmentation import CustomerSegmentation
        
        customer_segmentation = CustomerSegmentation(n_clusters=3)
        customer_segmentation.train(data=data)
        customer_segmentation.save_model('models/customer_segmentation.pkl')
        print("✅ Customer Segmentation trained") 
        
        # Train Fraud Detector (using fixed version)
        print("🛡️ Training Fraud Detector...")
        from src.ai_models.fraud_detector_fixed import FraudDetectorFixed
        
        fraud_detector = FraudDetectorFixed()
        fraud_detector.train(data=data)
        fraud_detector.save_model('models/fraud_detector.pkl')
        print("✅ Fraud Detector trained")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Quick setup main function"""
    print("🔧 QUICK SETUP - REAL DATA TRAINING")
    print("=" * 50)
    
    try:
        # Format the data properly
        if not create_properly_formatted_data():
            return False
            
        # Train models
        if not train_models_simple():
            return False
        
        print("\n" + "=" * 50)
        print("🎉 QUICK SETUP COMPLETED!")
        print("=" * 50)
        print("✅ Real data formatted and processed")
        print("✅ All AI models trained successfully")
        print("✅ Models saved and ready for use")
        print("\n🚀 Now run 'python main.py' to start the POS system!")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)