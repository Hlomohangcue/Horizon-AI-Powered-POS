#!/usr/bin/env python3
"""
Enhanced Sales Analytics Test
============================

Test the updated sales analytics with real product data and comprehensive analysis.
"""

import pandas as pd
from datetime import datetime, date

def format_currency(amount):
    """Format amount in Lesotho Maloti (LSL) currency"""
    return f"M {amount:,.2f}"

def test_enhanced_analytics():
    """Test enhanced sales analytics calculations"""
    print("🧪 TESTING ENHANCED SALES ANALYTICS")
    print("=" * 60)
    
    # Load data
    inventory = pd.read_csv('data/inventory.csv')
    transactions = pd.read_csv('data/transactions.csv')
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
    
    # Basic metrics (same as before)
    total_revenue = transactions['total_amount'].sum()
    total_transactions = len(transactions)
    avg_transaction = transactions['total_amount'].mean()
    total_items_sold = transactions['quantity'].sum()
    
    print("📊 CORE ANALYTICS METRICS:")
    print("-" * 40)
    print(f"💰 Total Revenue: {format_currency(total_revenue)}")
    print(f"🧾 Total Transactions: {total_transactions:,}")
    print(f"📊 Avg Transaction: {format_currency(avg_transaction)}")
    print(f"📦 Items Sold: {total_items_sold:,}")
    
    # Product analysis
    print(f"\n🏪 PRODUCT ANALYSIS:")
    print("-" * 40)
    
    # Top products by revenue
    top_products = transactions.groupby('product_name').agg({
        'quantity': 'sum',
        'total_amount': 'sum',
        'unit_price': 'mean'
    }).sort_values('total_amount', ascending=False)
    
    print(f"Top 5 Products by Revenue:")
    for i, (product, data) in enumerate(top_products.head(5).iterrows(), 1):
        print(f"{i}. {product}: {format_currency(data['total_amount'])} ({data['quantity']} units)")
    
    # Category analysis
    print(f"\n📊 CATEGORY PERFORMANCE:")
    print("-" * 40)
    category_performance = transactions.groupby('category').agg({
        'total_amount': 'sum',
        'quantity': 'sum',
        'product_name': 'nunique'
    }).sort_values('total_amount', ascending=False)
    
    for category, data in category_performance.iterrows():
        print(f"• {category}: {format_currency(data['total_amount'])} revenue, {data['quantity']} units, {data['product_name']} products")
    
    # Inventory vs Sales Analysis
    print(f"\n🔄 INVENTORY VS SALES ANALYSIS:")
    print("-" * 40)
    
    inventory_products = set(inventory['product_name'].str.lower().str.strip())
    transaction_products = set(transactions['product_name'].str.lower().str.strip())
    matching_products = inventory_products.intersection(transaction_products)
    
    print(f"📦 Products in Inventory: {len(inventory_products)}")
    print(f"🛒 Products Sold: {len(transaction_products)}")
    print(f"🎯 Matching Products: {len(matching_products)}")
    
    if len(matching_products) > 0:
        print(f"✅ Matching Products: {', '.join(list(matching_products)[:5])}")
    
    # Products sold but not in inventory
    products_without_inventory = transaction_products - inventory_products
    if len(products_without_inventory) > 0:
        print(f"⚠️  Products sold without inventory: {len(products_without_inventory)}")
        print(f"   Examples: {', '.join(list(products_without_inventory)[:3])}")
    
    # Unsold inventory
    unsold_products = inventory_products - transaction_products
    if len(unsold_products) > 0:
        print(f"📋 Unsold inventory items: {len(unsold_products)}")
        print(f"   Examples: {', '.join(list(unsold_products)[:3])}")
    
    # Date range analysis
    print(f"\n📅 DATE RANGE ANALYSIS:")
    print("-" * 40)
    min_date = transactions['transaction_date'].min().date()
    max_date = transactions['transaction_date'].max().date()
    date_range = (max_date - min_date).days
    
    print(f"📅 Date Range: {min_date} to {max_date} ({date_range} days)")
    print(f"📊 Daily Average Revenue: {format_currency(total_revenue / max(date_range, 1))}")
    print(f"📦 Daily Average Items Sold: {total_items_sold / max(date_range, 1):.1f}")
    
    # Performance summary
    print(f"\n🎯 PERFORMANCE SUMMARY:")
    print("-" * 40)
    best_day = transactions.groupby('transaction_date')['total_amount'].sum().idxmax()
    best_day_revenue = transactions.groupby('transaction_date')['total_amount'].sum().max()
    
    print(f"🏆 Best Sales Day: {best_day.date()} ({format_currency(best_day_revenue)})")
    print(f"💎 Highest Value Product: {top_products.index[0]} ({format_currency(top_products.iloc[0]['total_amount'])})")
    print(f"🔄 Most Sold Product: {top_products.sort_values('quantity', ascending=False).index[0]}")
    
    print(f"\n✅ ENHANCED ANALYTICS: READY")
    print("=" * 60)

if __name__ == "__main__":
    test_enhanced_analytics()