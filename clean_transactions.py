#!/usr/bin/env python3
"""
Remove Electronics/Phone Products from Transactions
===================================================

This script removes Samsung Galaxy S23, iPhone 14, MacBook Pro, and 
Wireless Headphones from the transaction data to clean up the sales records.
"""

import pandas as pd
from datetime import datetime

def clean_transactions():
    """Remove electronics/phone products from transactions"""
    
    print("REMOVING ELECTRONICS/PHONE PRODUCTS FROM TRANSACTIONS")
    print("=" * 60)
    
    # Load current transactions
    transactions = pd.read_csv('data/transactions.csv')
    
    # Create backup first
    backup_filename = f"data/transactions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    transactions.to_csv(backup_filename, index=False)
    print(f"Backup created: {backup_filename}")
    
    # Define electronics/phone products to remove
    electronics_products = [
        'MacBook Pro',
        'Samsung Galaxy S23', 
        'Wireless Headphones',
        'iPhone 14'
    ]
    
    # Filter out electronics products
    cleaned_transactions = transactions[~transactions['product_name'].isin(electronics_products)]
    
    print(f"\nCLEANING RESULTS:")
    print("-" * 30)
    print(f"Original transactions: {len(transactions)}")
    print(f"Electronics transactions removed: {len(transactions) - len(cleaned_transactions)}")
    print(f"Remaining transactions: {len(cleaned_transactions)}")
    
    # Calculate revenue impact
    original_revenue = transactions['total_amount'].sum()
    removed_revenue = transactions[transactions['product_name'].isin(electronics_products)]['total_amount'].sum()
    remaining_revenue = cleaned_transactions['total_amount'].sum()
    
    print(f"\nREVENUE IMPACT:")
    print("-" * 20)
    print(f"Original revenue: M {original_revenue:,.2f}")
    print(f"Removed revenue: M {removed_revenue:,.2f}")
    print(f"Remaining revenue: M {remaining_revenue:,.2f}")
    
    print(f"\nREMAINING PRODUCTS:")
    print("-" * 25)
    remaining_products = cleaned_transactions.groupby(['product_name', 'category']).agg({
        'quantity': 'sum',
        'total_amount': 'sum'
    }).round(2)
    
    for (product, category), data in remaining_products.iterrows():
        print(f"• {product} ({category}): {data['quantity']} units, M {data['total_amount']:,.2f}")
    
    # Save cleaned transactions
    cleaned_transactions.to_csv('data/transactions.csv', index=False)
    print(f"\nCleaned transactions saved to: data/transactions.csv")
    print(f"Electronics products successfully removed!")
    
    # Verify the save
    verification = pd.read_csv('data/transactions.csv')
    print(f"\nVERIFICATION:")
    print(f"New file contains: {len(verification)} transactions")
    print(f"New total revenue: M {verification['total_amount'].sum():,.2f}")
    
    # Update customer and transaction IDs to be sequential
    verification = verification.reset_index(drop=True)
    verification['transaction_id'] = [f"TXN_{i+1:03d}" for i in range(len(verification))]
    verification.to_csv('data/transactions.csv', index=False)
    
    print(f"Transaction IDs updated to be sequential (TXN_001 to TXN_{len(verification):03d})")
    
    return verification

if __name__ == "__main__":
    cleaned_data = clean_transactions()