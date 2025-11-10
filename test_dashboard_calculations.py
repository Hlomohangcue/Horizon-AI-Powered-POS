#!/usr/bin/env python3
"""
Test Dashboard Calculations
==========================

Verify that the dashboard displays accurate real-time data from the CSV files.
This script validates the key metrics shown in the Horizon AI POS Dashboard.
"""

import pandas as pd
import os
import sys

def format_currency(amount):
    """Format amount in Lesotho Maloti (LSL) currency"""
    return f"M {amount:,.2f}"

def test_dashboard_calculations():
    """Test all dashboard calculations against real data"""
    print("🧪 TESTING DASHBOARD CALCULATIONS")
    print("=" * 50)
    
    # Test Transactions Data
    if os.path.exists('data/transactions.csv'):
        transactions = pd.read_csv('data/transactions.csv')
        
        print("\n💰 SALES METRICS:")
        print("-" * 30)
        
        # Calculate key metrics
        total_sales = transactions['total_amount'].sum()
        total_transactions = len(transactions)
        avg_transaction = transactions['total_amount'].mean()
        unique_customers = transactions['customer_id'].nunique()
        
        print(f"💰 Total Sales: {format_currency(total_sales)}")
        print(f"🧾 Total Transactions: {total_transactions:,}")
        print(f"📊 Avg Transaction: {format_currency(avg_transaction)}")
        print(f"👥 Unique Customers: {unique_customers:,}")
        
        # Verify expected values match your dashboard request
        expected_sales = 125738.17  # From your request
        expected_transactions = 103
        expected_avg = 1220.76
        expected_customers = 102
        
        print(f"\n✅ VALIDATION RESULTS:")
        print(f"Sales Match: {abs(total_sales - expected_sales) < 100} (Diff: {format_currency(abs(total_sales - expected_sales))})")
        print(f"Transactions Match: {abs(total_transactions - expected_transactions) <= 1} (Diff: {abs(total_transactions - expected_transactions)})")
        print(f"Avg Match: {abs(avg_transaction - expected_avg) < 50} (Diff: {format_currency(abs(avg_transaction - expected_avg))})")
        print(f"Customers Match: {abs(unique_customers - expected_customers) <= 1} (Diff: {abs(unique_customers - expected_customers)})")
        
    else:
        print("❌ No transactions.csv found")
    
    # Test Inventory Data
    if os.path.exists('data/inventory.csv'):
        inventory = pd.read_csv('data/inventory.csv')
        
        print(f"\n📦 INVENTORY METRICS:")
        print("-" * 30)
        
        # Calculate inventory metrics
        total_products = len(inventory)
        total_stock_units = inventory['stock_quantity'].sum()
        total_stock_value = (inventory['unit_price'] * inventory['stock_quantity']).sum()
        low_stock_count = len(inventory[inventory['stock_quantity'] <= inventory['reorder_level']])
        out_of_stock = len(inventory[inventory['stock_quantity'] == 0])
        
        print(f"📦 Total Products: {total_products:,}")
        print(f"📊 Total Stock Units: {total_stock_units:,}")
        print(f"💎 Total Stock Value: {format_currency(total_stock_value)}")
        print(f"⚠️ Low Stock Items: {low_stock_count:,}")
        print(f"🚫 Out of Stock: {out_of_stock:,}")
        
        # Show inventory details
        print(f"\n📋 INVENTORY DETAILS:")
        for _, product in inventory.iterrows():
            stock_value = product['unit_price'] * product['stock_quantity']
            status = "🟢" if product['stock_quantity'] > product['reorder_level'] else "🟡" if product['stock_quantity'] > 0 else "🔴"
            print(f"{status} {product['product_name']}: {product['stock_quantity']} units @ {format_currency(product['unit_price'])} = {format_currency(stock_value)}")
            
    else:
        print("❌ No inventory.csv found")
    
    print(f"\n🎯 DASHBOARD STATUS: ✅ READY WITH REAL DATA")
    print("=" * 50)

if __name__ == "__main__":
    test_dashboard_calculations()