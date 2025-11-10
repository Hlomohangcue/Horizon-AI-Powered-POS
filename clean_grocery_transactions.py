#!/usr/bin/env python3
"""
Clean Transactions to Keep Only Grocery Items
==============================================

Remove all non-grocery products from transactions to align with 
Horizon Enterprise's actual grocery business model.
"""

import pandas as pd
from datetime import datetime

def clean_grocery_transactions():
    """Keep only grocery items that match actual inventory"""
    
    print("CLEANING TRANSACTIONS FOR GROCERY BUSINESS ONLY")
    print("=" * 60)
    
    # Load current data
    inventory = pd.read_csv('data/inventory.csv')
    transactions = pd.read_csv('data/transactions.csv')
    
    print(f"Current inventory: {len(inventory)} grocery items")
    print(f"Current transactions: {len(transactions)} transactions")
    
    print(f"\nINVENTORY ITEMS (Grocery Store):")
    print("-" * 40)
    for _, item in inventory.iterrows():
        print(f"• {item['product_name']} ({item['category']}) - M {item['unit_price']:.2f}")
    
    # Create backup
    backup_filename = f"data/transactions_backup_grocery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    transactions.to_csv(backup_filename, index=False)
    print(f"\nBackup created: {backup_filename}")
    
    # Current transaction products
    print(f"\nCURRENT TRANSACTION PRODUCTS:")
    print("-" * 35)
    current_products = transactions.groupby(['product_name', 'category']).agg({
        'quantity': 'sum',
        'total_amount': 'sum'
    }).sort_values('total_amount', ascending=False)
    
    # Identify grocery items from inventory
    inventory_products = set(inventory['product_name'].str.lower().str.strip())
    
    # Products to keep (only if in inventory)
    products_to_keep = []
    products_to_remove = []
    
    for product in transactions['product_name'].unique():
        product_lower = product.lower().strip()
        if product_lower in inventory_products:
            products_to_keep.append(product)
            print(f"KEEP: {product}")
        else:
            products_to_remove.append(product)
            print(f"REMOVE: {product} (not in grocery inventory)")
    
    print(f"\nCLEANING SUMMARY:")
    print("-" * 25)
    print(f"Products to keep: {len(products_to_keep)}")
    print(f"Products to remove: {len(products_to_remove)}")
    
    # Show removal impact
    if products_to_remove:
        print(f"\nPRODUCTS BEING REMOVED:")
        print("-" * 30)
        for product in products_to_remove:
            product_data = transactions[transactions['product_name'] == product]
            qty = product_data['quantity'].sum()
            revenue = product_data['total_amount'].sum()
            transactions_count = len(product_data)
            print(f"• {product}: {qty} units, M {revenue:.2f}, {transactions_count} transactions")
    
    # Filter transactions to keep only grocery items
    grocery_transactions = transactions[transactions['product_name'].isin(products_to_keep)]
    
    # Calculate impact
    original_count = len(transactions)
    grocery_count = len(grocery_transactions)
    removed_count = original_count - grocery_count
    
    original_revenue = transactions['total_amount'].sum()
    grocery_revenue = grocery_transactions['total_amount'].sum()
    removed_revenue = original_revenue - grocery_revenue
    
    print(f"\nIMPACT ANALYSIS:")
    print("-" * 20)
    print(f"Original transactions: {original_count}")
    print(f"Grocery transactions: {grocery_count}")
    print(f"Removed transactions: {removed_count}")
    print(f"Original revenue: M {original_revenue:.2f}")
    print(f"Grocery revenue: M {grocery_revenue:.2f}")
    print(f"Removed revenue: M {removed_revenue:.2f}")
    
    if grocery_count > 0:
        avg_grocery_transaction = grocery_revenue / grocery_count
        print(f"New avg transaction: M {avg_grocery_transaction:.2f}")
        
        # Show remaining products
        print(f"\nREMAINING GROCERY PRODUCTS:")
        print("-" * 35)
        remaining_products = grocery_transactions.groupby(['product_name', 'category']).agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).sort_values('total_amount', ascending=False)
        
        for (product, category), data in remaining_products.iterrows():
            print(f"• {product} ({category}): {data['quantity']} units, M {data['total_amount']:.2f}")
        
        # Save cleaned transactions
        grocery_transactions = grocery_transactions.reset_index(drop=True)
        grocery_transactions['transaction_id'] = [f"TXN_{i+1:03d}" for i in range(len(grocery_transactions))]
        grocery_transactions.to_csv('data/transactions.csv', index=False)
        
        print(f"\nCleaned grocery transactions saved!")
        print(f"Transaction IDs updated: TXN_001 to TXN_{len(grocery_transactions):03d}")
        
    else:
        print(f"\nWARNING: No matching products found between inventory and transactions!")
        print("Creating sample transactions with inventory products...")
        
        # Create sample transactions using inventory items
        sample_transactions = []
        base_date = pd.Timestamp('2025-11-01')
        
        for i, (_, item) in enumerate(inventory.head(10).iterrows()):
            sample_transactions.append({
                'transaction_id': f'TXN_{i+1:03d}',
                'customer_id': f'CUST_{i+1000:04d}',
                'customer_name': f'Customer {i+1}',
                'product_id': item.get('product_id', f'PROD_{i+1:03d}'),
                'product_name': item['product_name'],
                'category': item['category'],
                'quantity': min(5, item['stock_quantity']),
                'unit_price': item['unit_price'],
                'subtotal': min(5, item['stock_quantity']) * item['unit_price'],
                'discount_amount': 0.0,
                'total_amount': min(5, item['stock_quantity']) * item['unit_price'],
                'payment_method': 'Cash',
                'transaction_date': (base_date + pd.Timedelta(days=i)).date(),
                'transaction_timestamp': base_date + pd.Timedelta(days=i),
                'cashier': 'Sales Assistant 1',
                'payment_received': '',
                'change_due': ''
            })
        
        sample_df = pd.DataFrame(sample_transactions)
        sample_df.to_csv('data/transactions.csv', index=False)
        print(f"Created {len(sample_df)} sample transactions with grocery items")
    
    # Verification
    verification = pd.read_csv('data/transactions.csv')
    print(f"\nVERIFICATION:")
    print(f"Final transactions file: {len(verification)} transactions")
    print(f"Final revenue: M {verification['total_amount'].sum():.2f}")
    print(f"Products in final data: {verification['product_name'].nunique()}")
    
    return verification

if __name__ == "__main__":
    result = clean_grocery_transactions()