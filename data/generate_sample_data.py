"""
Sample Data Generator for Horizon AI-Powered POS System
======================================================

This module generates synthetic transaction data for training and testing
the AI models in the POS system.

Author: [Your Name]
Course: AI for Software Engineering
Date: November 8, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_sample_data(n_transactions=5000, n_customers=500, save_to_file=True):
    """
    Generate synthetic transaction data for the POS system
    
    Args:
        n_transactions (int): Number of transactions to generate
        n_customers (int): Number of unique customers
        save_to_file (bool): Whether to save data to CSV files
        
    Returns:
        pd.DataFrame: Generated transaction data
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating {n_transactions} transactions for {n_customers} customers...")
    
    # Product catalog
    products = {
        'ELEC001': {'name': 'Smartphone', 'category': 'Electronics', 'price': 599.99},
        'ELEC002': {'name': 'Laptop', 'category': 'Electronics', 'price': 999.99},
        'ELEC003': {'name': 'Tablet', 'category': 'Electronics', 'price': 399.99},
        'ELEC004': {'name': 'Headphones', 'category': 'Electronics', 'price': 149.99},
        'CLOTH001': {'name': 'T-Shirt', 'category': 'Clothing', 'price': 29.99},
        'CLOTH002': {'name': 'Jeans', 'category': 'Clothing', 'price': 79.99},
        'CLOTH003': {'name': 'Sneakers', 'category': 'Clothing', 'price': 89.99},
        'CLOTH004': {'name': 'Jacket', 'category': 'Clothing', 'price': 129.99},
        'FOOD001': {'name': 'Coffee', 'category': 'Food', 'price': 4.99},
        'FOOD002': {'name': 'Sandwich', 'category': 'Food', 'price': 8.99},
        'FOOD003': {'name': 'Salad', 'category': 'Food', 'price': 12.99},
        'FOOD004': {'name': 'Pizza', 'category': 'Food', 'price': 18.99},
        'JEWEL001': {'name': 'Gold Ring', 'category': 'Jewelry', 'price': 1299.99},
        'JEWEL002': {'name': 'Silver Necklace', 'category': 'Jewelry', 'price': 199.99},
        'BOOK001': {'name': 'AI Textbook', 'category': 'Books', 'price': 89.99},
        'BOOK002': {'name': 'Novel', 'category': 'Books', 'price': 19.99}
    }
    
    # Customer segments with different behaviors
    customer_segments = {
        'Budget': {'price_sensitivity': 0.8, 'frequency': 0.3, 'categories': ['Food', 'Books']},
        'Regular': {'price_sensitivity': 0.5, 'frequency': 0.6, 'categories': ['Clothing', 'Food', 'Electronics']},
        'Premium': {'price_sensitivity': 0.2, 'frequency': 0.4, 'categories': ['Electronics', 'Jewelry']},
        'Frequent': {'price_sensitivity': 0.6, 'frequency': 0.9, 'categories': ['Food', 'Clothing']}
    }
    
    # Generate customer profiles
    customers = {}
    for i in range(n_customers):
        customer_id = f"CUST_{i+1:05d}"
        segment = random.choice(list(customer_segments.keys()))
        customers[customer_id] = {
            'segment': segment,
            'join_date': datetime.now() - timedelta(days=random.randint(30, 730)),
            'loyalty_score': random.uniform(0, 1)
        }
    
    # Generate transactions
    transactions = []
    
    # Time period: last 365 days
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_transactions):
        # Select random customer
        customer_id = random.choice(list(customers.keys()))
        customer = customers[customer_id]
        
        # Generate transaction timestamp
        # More transactions during business hours and recent dates
        days_ago = int(np.random.exponential(30))  # Exponential distribution favoring recent dates
        days_ago = min(days_ago, 365)
        
        # Fixed probability distribution that sums to exactly 1.0
        hour_probs = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5: Very low
            0.02, 0.03, 0.05, 0.08, 0.10, 0.12,  # 6-11: Morning ramp up
            0.15, 0.15, 0.15, 0.12, 0.10, 0.08,  # 12-17: Peak hours
            0.06, 0.04, 0.03, 0.02, 0.01, 0.01   # 18-23: Evening decline
        ])
        # Normalize to ensure sum equals 1.0
        hour_probs = hour_probs / hour_probs.sum()
        hour = np.random.choice(range(24), p=hour_probs)
        
        transaction_time = start_date + timedelta(
            days=365-days_ago,
            hours=hour,
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Select product based on customer segment
        segment_info = customer_segments[customer['segment']]
        preferred_categories = segment_info['categories']
        
        # Filter products by preferred categories
        available_products = {k: v for k, v in products.items() 
                            if v['category'] in preferred_categories}
        
        if not available_products:
            available_products = products  # Fallback to all products
        
        product_id = random.choice(list(available_products.keys()))
        product = products[product_id]
        
        # Quantity (influenced by product type and customer segment)
        if product['category'] == 'Food':
            quantity = random.randint(1, 3)
        elif product['category'] in ['Electronics', 'Jewelry']:
            quantity = 1 if random.random() < 0.9 else 2  # Usually buy 1
        else:
            quantity = random.randint(1, 5)
        
        # Apply price variations
        base_price = product['price']
        price_variation = random.uniform(0.9, 1.1)  # ±10% price variation
        unit_price = base_price * price_variation
        
        # Apply customer loyalty discount
        if customer['loyalty_score'] > 0.7:
            unit_price *= 0.95  # 5% loyalty discount
        
        total_amount = unit_price * quantity
        
        # Payment method (influenced by amount and customer segment)
        if total_amount < 20:
            payment_method = random.choice(['cash', 'cash', 'debit_card'])
        elif total_amount < 100:
            payment_method = random.choice(['cash', 'debit_card', 'credit_card'])
        else:
            payment_method = random.choice(['credit_card', 'credit_card', 'online_payment'])
        
        # Store location
        store_location = random.choice(['Store_A', 'Store_B', 'Store_C', 'Store_D'])
        
        # Create transaction record
        transaction = {
            'transaction_id': f"TXN_{i+1:06d}",
            'customer_id': customer_id,
            'product_id': product_id,
            'product_name': product['name'],
            'product_category': product['category'],
            'quantity': quantity,
            'unit_price': round(unit_price, 2),
            'total_amount': round(total_amount, 2),
            'payment_method': payment_method,
            'store_location': store_location,
            'transaction_timestamp': transaction_time,
            'transaction_date': transaction_time.date(),
            'customer_segment': customer['segment'],
            'customer_loyalty_score': customer['loyalty_score']
        }
        
        transactions.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Sort by timestamp
    df = df.sort_values('transaction_timestamp').reset_index(drop=True)
    
    # Add some fraud cases (5% of transactions)
    fraud_indices = random.sample(range(len(df)), int(0.05 * len(df)))
    df['is_fraud'] = 0
    df.loc[fraud_indices, 'is_fraud'] = 1
    
    # Make fraud cases more obvious
    for idx in fraud_indices:
        # High amounts
        if random.random() < 0.3:
            df.loc[idx, 'total_amount'] *= random.uniform(3, 10)
        
        # Night transactions
        if random.random() < 0.4:
            night_hour = random.choice([1, 2, 3, 4, 5])
            original_time = df.loc[idx, 'transaction_timestamp']
            df.loc[idx, 'transaction_timestamp'] = original_time.replace(hour=night_hour)
        
        # Round amounts
        if random.random() < 0.3:
            df.loc[idx, 'total_amount'] = round(df.loc[idx, 'total_amount'] / 100) * 100
    
    print(f"Generated {len(df)} transactions")
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    
    # Save to files if requested
    if save_to_file:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save main transaction data
        main_file = os.path.join(data_dir, 'transactions.csv')
        df.to_csv(main_file, index=False)
        print(f"Transaction data saved to: {main_file}")
        
        # Save customer data
        customer_df = pd.DataFrame([
            {
                'customer_id': cid,
                'segment': info['segment'],
                'join_date': info['join_date'],
                'loyalty_score': info['loyalty_score']
            }
            for cid, info in customers.items()
        ])
        
        customer_file = os.path.join(data_dir, 'customers.csv')
        customer_df.to_csv(customer_file, index=False)
        print(f"Customer data saved to: {customer_file}")
        
        # Save product catalog
        product_df = pd.DataFrame([
            {
                'product_id': pid,
                'name': info['name'],
                'category': info['category'],
                'price': info['price']
            }
            for pid, info in products.items()
        ])
        
        product_file = os.path.join(data_dir, 'products.csv')
        product_df.to_csv(product_file, index=False)
        print(f"Product catalog saved to: {product_file}")
    
    return df

def generate_training_datasets():
    """Generate separate datasets for training different models"""
    
    print("Generating training datasets for AI models...")
    
    # Generate main transaction data
    transaction_data = generate_sample_data(n_transactions=10000, n_customers=1000, save_to_file=False)
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Sales prediction dataset (aggregate by date)
    sales_data = transaction_data.groupby('transaction_date').agg({
        'total_amount': 'sum',
        'quantity': 'sum',
        'transaction_id': 'count',
        'customer_id': 'nunique',
        'product_category': lambda x: x.mode()[0] if not x.empty else 'Unknown'
    }).rename(columns={'transaction_id': 'transaction_count', 'customer_id': 'unique_customers'})
    
    sales_data = sales_data.reset_index()
    sales_file = os.path.join(data_dir, 'sales_history.csv')
    sales_data.to_csv(sales_file, index=False)
    print(f"Sales prediction data saved to: {sales_file}")
    
    # Customer segmentation dataset
    customer_features = transaction_data.groupby('customer_id').agg({
        'transaction_timestamp': 'max',
        'total_amount': ['count', 'sum', 'mean'],
        'quantity': ['sum', 'mean'],
        'product_category': 'nunique'
    })
    
    customer_features.columns = ['last_transaction', 'frequency', 'monetary_total', 'monetary_avg', 'total_quantity', 'avg_quantity', 'product_categories']
    customer_features['recency'] = (datetime.now() - customer_features['last_transaction']).dt.days
    customer_features = customer_features.reset_index()
    
    customer_file = os.path.join(data_dir, 'customer_behavior.csv')
    customer_features.to_csv(customer_file, index=False)
    print(f"Customer segmentation data saved to: {customer_file}")
    
    # Fraud detection dataset (full transaction data with fraud labels)
    fraud_file = os.path.join(data_dir, 'fraud_training.csv')
    transaction_data.to_csv(fraud_file, index=False)
    print(f"Fraud detection data saved to: {fraud_file}")
    
    print("All training datasets generated successfully!")

if __name__ == "__main__":
    # Generate sample data
    generate_sample_data()
    
    # Generate specialized training datasets
    generate_training_datasets()
    
    print("\n✅ Data generation completed!")
    print("You can now train the AI models using the generated datasets.")