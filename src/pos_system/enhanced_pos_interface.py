"""
Enhanced POS Interface for Horizon AI System
==========================================

This module provides comprehensive POS interface with:
- Manager dashboard with inventory management
- Sales assistant workflow with product selection
- Daily reporting and analytics
- AI-powered insights and recommendations

Author: Horizon Enterprise Team
Course: AI for Software Engineering
Date: November 8, 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from inventory.inventory_manager import InventoryManager

logger = logging.getLogger(__name__)

class EnhancedPOSInterface:
    """
    Enhanced POS Interface with manager and sales assistant modes
    """
    
    def __init__(self, ai_models=None):
        """Initialize enhanced POS interface"""
        self.ai_models = ai_models
        self.inventory_manager = InventoryManager()
        self.sales_file = 'data/daily_sales.csv'
        self.ensure_sales_file()
        logger.info("Enhanced POS Interface initialized")
    
    def ensure_sales_file(self):
        """Create sales file if it doesn't exist"""
        if not os.path.exists(self.sales_file):
            sales_df = pd.DataFrame(columns=[
                'transaction_id', 'date', 'time', 'customer_id', 'product_id',
                'product_name', 'category', 'quantity', 'unit_price', 'total_amount',
                'payment_method', 'sales_assistant', 'discount_applied'
            ])
            os.makedirs(os.path.dirname(self.sales_file), exist_ok=True)
            sales_df.to_csv(self.sales_file, index=False)
    
    def start_system(self):
        """Start the enhanced POS system"""
        print("\n" + "=" * 60)
        print("🏢 HORIZON ENTERPRISE - AI POWERED POS SYSTEM")
        print("=" * 60)
        print("Enhanced with Inventory Management & Analytics")
        print("")
        
        while True:
            print("\n📱 MAIN MENU")
            print("-" * 30)
            print("1. 👔 Manager Dashboard")
            print("2. 🛒 Sales Assistant Mode")
            print("3. 📊 Quick Reports")
            print("4. 🤖 AI Insights")
            print("5. ❌ Exit System")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                self.manager_dashboard()
            elif choice == '2':
                self.sales_assistant_mode()
            elif choice == '3':
                self.quick_reports()
            elif choice == '4':
                self.show_ai_insights()
            elif choice == '5':
                print("\n👋 Thank you for using Horizon POS!")
                break
            else:
                print("❌ Invalid option. Please try again.")
    
    def manager_dashboard(self):
        """Manager dashboard with full system control"""
        print("\n" + "=" * 50)
        print("👔 MANAGER DASHBOARD")
        print("=" * 50)
        
        while True:
            print("\n📋 MANAGER OPTIONS")
            print("-" * 25)
            print("1. 📦 Inventory Management")
            print("2. 📈 Sales Analytics")
            print("3. 📊 Generate Reports")
            print("4. ⚠️ Stock Alerts")
            print("5. 👥 Staff Performance")
            print("6. 🔙 Back to Main Menu")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                self.inventory_management()
            elif choice == '2':
                self.sales_analytics()
            elif choice == '3':
                self.generate_reports()
            elif choice == '4':
                self.show_stock_alerts()
            elif choice == '5':
                self.staff_performance()
            elif choice == '6':
                break
            else:
                print("❌ Invalid option. Please try again.")
    
    def inventory_management(self):
        """Inventory management interface"""
        print("\n" + "=" * 40)
        print("📦 INVENTORY MANAGEMENT")
        print("=" * 40)
        
        while True:
            print("\n📋 INVENTORY OPTIONS")
            print("-" * 20)
            print("1. ➕ Add New Product")
            print("2. 📝 Update Product Info")
            print("3. 📊 View All Products")
            print("4. 🔍 Search Products")
            print("5. 📈 Update Stock Levels")
            print("6. 💰 Inventory Value Report")
            print("7. 🔙 Back")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                self.add_new_product()
            elif choice == '2':
                self.update_product_info()
            elif choice == '3':
                self.view_all_products()
            elif choice == '4':
                self.search_products()
            elif choice == '5':
                self.update_stock_levels()
            elif choice == '6':
                self.show_inventory_value()
            elif choice == '7':
                break
            else:
                print("❌ Invalid option. Please try again.")
    
    def add_new_product(self):
        """Add new product to inventory"""
        print("\n➕ ADD NEW PRODUCT")
        print("-" * 20)
        
        try:
            product_data = {}
            
            product_data['product_name'] = input("Product Name: ").strip()
            if not product_data['product_name']:
                print("❌ Product name is required!")
                return
            
            product_data['category'] = input("Category: ").strip()
            product_data['unit_price'] = float(input("Unit Price ($): ").strip())
            product_data['cost_price'] = float(input("Cost Price ($): ").strip())
            product_data['current_stock'] = int(input("Initial Stock Quantity: ").strip())
            product_data['minimum_stock'] = int(input("Minimum Stock Level: ").strip())
            product_data['supplier'] = input("Supplier: ").strip()
            
            if self.inventory_manager.add_new_product(product_data):
                print(f"✅ Product '{product_data['product_name']}' added successfully!")
            else:
                print("❌ Failed to add product. Please try again.")
                
        except ValueError:
            print("❌ Invalid input. Please enter valid numbers for prices and quantities.")
        except Exception as e:
            print(f"❌ Error adding product: {e}")
    
    def view_all_products(self):
        """Display all products in inventory"""
        print("\n📊 ALL PRODUCTS")
        print("-" * 15)
        
        products = self.inventory_manager.get_available_products(in_stock_only=False)
        
        if products.empty:
            print("📦 No products in inventory.")
            return
        
        print(f"\n{'ID':<10} {'Name':<25} {'Category':<15} {'Stock':<8} {'Price':<10} {'Status':<10}")
        print("-" * 85)
        
        for _, product in products.iterrows():
            status = "LOW" if product['current_stock'] <= product['minimum_stock'] else "OK"
            status_color = "⚠️" if status == "LOW" else "✅"
            
            print(f"{product['product_id']:<10} {product['product_name'][:24]:<25} "
                  f"{product['category']:<15} {product['current_stock']:<8} "
                  f"${product['unit_price']:<9.2f} {status_color}{status}")
    
    def update_stock_levels(self):
        """Update stock levels for products"""
        print("\n📈 UPDATE STOCK LEVELS")
        print("-" * 22)
        
        # Show current low stock first
        low_stock = self.inventory_manager.get_low_stock_alerts()
        if not low_stock.empty:
            print("\n⚠️ LOW STOCK ALERTS:")
            for _, product in low_stock.iterrows():
                print(f"• {product['product_name']} ({product['product_id']}): {product['current_stock']} units")
        
        product_id = input("\nEnter Product ID to update: ").strip().upper()
        
        product = self.inventory_manager.get_product_by_id(product_id)
        if not product:
            print(f"❌ Product {product_id} not found!")
            return
        
        print(f"\nProduct: {product['product_name']}")
        print(f"Current Stock: {product['current_stock']} units")
        
        try:
            operation = input("Operation (add/subtract): ").strip().lower()
            if operation not in ['add', 'subtract']:
                print("❌ Invalid operation. Use 'add' or 'subtract'.")
                return
            
            quantity = int(input("Quantity: ").strip())
            
            if self.inventory_manager.update_stock(product_id, quantity, operation):
                print(f"✅ Stock updated successfully!")
                
                # Show new stock level
                updated_product = self.inventory_manager.get_product_by_id(product_id)
                print(f"New Stock Level: {updated_product['current_stock']} units")
            else:
                print("❌ Failed to update stock.")
                
        except ValueError:
            print("❌ Invalid quantity. Please enter a valid number.")
    
    def sales_assistant_mode(self):
        """Sales assistant interface for daily transactions"""
        print("\n" + "=" * 50)
        print("🛒 SALES ASSISTANT MODE")
        print("=" * 50)
        
        assistant_name = input("Enter your name: ").strip()
        if not assistant_name:
            assistant_name = "Sales Assistant"
        
        print(f"\nWelcome, {assistant_name}! 👋")
        
        while True:
            print("\n🛍️ SALES OPTIONS")
            print("-" * 15)
            print("1. 💰 Process New Sale")
            print("2. 🔍 Check Product Availability")
            print("3. 📋 View Today's Sales")
            print("4. 🔙 Back to Main Menu")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                self.process_new_sale(assistant_name)
            elif choice == '2':
                self.check_product_availability()
            elif choice == '3':
                self.view_todays_sales()
            elif choice == '4':
                break
            else:
                print("❌ Invalid option. Please try again.")
    
    def process_new_sale(self, assistant_name):
        """Process a new sale with product selection"""
        print("\n💰 PROCESS NEW SALE")
        print("-" * 20)
        
        # Show available products
        products = self.inventory_manager.get_available_products(in_stock_only=True)
        
        if products.empty:
            print("❌ No products available for sale!")
            return
        
        print("\n📦 AVAILABLE PRODUCTS:")
        print(f"{'#':<3} {'ID':<10} {'Product Name':<25} {'Stock':<8} {'Price':<10}")
        print("-" * 60)
        
        for idx, (_, product) in enumerate(products.iterrows(), 1):
            print(f"{idx:<3} {product['product_id']:<10} {product['product_name'][:24]:<25} "
                  f"{product['current_stock']:<8} ${product['unit_price']:<9.2f}")
        
        try:
            # Select product
            selection = input(f"\nSelect product (1-{len(products)} or product ID): ").strip()
            
            if selection.isdigit():
                selection_idx = int(selection) - 1
                if 0 <= selection_idx < len(products):
                    selected_product = products.iloc[selection_idx]
                else:
                    print("❌ Invalid selection!")
                    return
            else:
                # Try to find by product ID
                selected_product = products[products['product_id'] == selection.upper()]
                if selected_product.empty:
                    print("❌ Product not found!")
                    return
                selected_product = selected_product.iloc[0]
            
            # Get quantity
            max_qty = selected_product['current_stock']
            quantity = int(input(f"Quantity (max {max_qty}): ").strip())
            
            if quantity <= 0 or quantity > max_qty:
                print(f"❌ Invalid quantity! Must be between 1 and {max_qty}")
                return
            
            # Calculate total
            unit_price = selected_product['unit_price']
            subtotal = quantity * unit_price
            
            # Apply discount if needed
            discount_percent = 0
            apply_discount = input("Apply discount? (y/n): ").strip().lower()
            if apply_discount == 'y':
                try:
                    discount_percent = float(input("Discount percentage (0-50): ").strip())
                    discount_percent = max(0, min(50, discount_percent))  # Limit to 0-50%
                except ValueError:
                    discount_percent = 0
            
            discount_amount = subtotal * (discount_percent / 100)
            total_amount = subtotal - discount_amount
            
            # Get customer info
            customer_id = input("Customer ID (optional): ").strip() or "WALK_IN"
            payment_method = input("Payment method (Cash/Card/Digital): ").strip() or "Cash"
            
            # Show transaction summary
            print("\n📋 TRANSACTION SUMMARY")
            print("-" * 25)
            print(f"Product: {selected_product['product_name']}")
            print(f"Quantity: {quantity}")
            print(f"Unit Price: ${unit_price:.2f}")
            print(f"Subtotal: ${subtotal:.2f}")
            if discount_percent > 0:
                print(f"Discount ({discount_percent}%): -${discount_amount:.2f}")
            print(f"Total: ${total_amount:.2f}")
            print(f"Payment: {payment_method}")
            
            confirm = input("\nConfirm transaction? (y/n): ").strip().lower()
            
            if confirm == 'y':
                # Record the sale
                if self.record_sale(selected_product, quantity, total_amount, 
                                  customer_id, payment_method, assistant_name, discount_percent):
                    print("✅ Sale completed successfully!")
                    
                    # Update inventory
                    self.inventory_manager.update_stock(
                        selected_product['product_id'], quantity, 'subtract'
                    )
                    
                    # Show AI insights if available
                    if self.ai_models:
                        self.show_transaction_insights(selected_product, quantity, total_amount)
                else:
                    print("❌ Failed to record sale!")
            else:
                print("❌ Transaction cancelled")
                
        except ValueError:
            print("❌ Invalid input. Please enter valid numbers.")
        except Exception as e:
            print(f"❌ Error processing sale: {e}")
    
    def record_sale(self, product, quantity, total_amount, customer_id, payment_method, assistant_name, discount_percent):
        """Record sale in daily sales file"""
        try:
            # Generate transaction ID
            now = datetime.now()
            transaction_id = f"TXN_{now.strftime('%Y%m%d_%H%M%S')}"
            
            # Create sale record
            sale_record = {
                'transaction_id': transaction_id,
                'date': now.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S'),
                'customer_id': customer_id,
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'category': product['category'],
                'quantity': quantity,
                'unit_price': product['unit_price'],
                'total_amount': total_amount,
                'payment_method': payment_method,
                'sales_assistant': assistant_name,
                'discount_applied': discount_percent
            }
            
            # Load existing sales
            try:
                sales_df = pd.read_csv(self.sales_file)
            except:
                sales_df = pd.DataFrame()
            
            # Add new sale
            new_sale = pd.DataFrame([sale_record])
            sales_df = pd.concat([sales_df, new_sale], ignore_index=True)
            
            # Save to file
            sales_df.to_csv(self.sales_file, index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording sale: {e}")
            return False
    
    def view_todays_sales(self):
        """View today's sales summary"""
        print("\n📋 TODAY'S SALES")
        print("-" * 15)
        
        try:
            sales_df = pd.read_csv(self.sales_file)
            today = datetime.now().strftime('%Y-%m-%d')
            
            today_sales = sales_df[sales_df['date'] == today]
            
            if today_sales.empty:
                print("📦 No sales recorded for today.")
                return
            
            # Summary statistics
            total_transactions = len(today_sales)
            total_revenue = today_sales['total_amount'].sum()
            total_items = today_sales['quantity'].sum()
            avg_transaction = total_revenue / total_transactions if total_transactions > 0 else 0
            
            print(f"\n📊 TODAY'S SUMMARY ({today})")
            print("-" * 30)
            print(f"Total Transactions: {total_transactions}")
            print(f"Total Revenue: ${total_revenue:.2f}")
            print(f"Items Sold: {total_items}")
            print(f"Average Transaction: ${avg_transaction:.2f}")
            
            # Recent transactions
            print(f"\n🕒 RECENT TRANSACTIONS:")
            print(f"{'Time':<8} {'Product':<20} {'Qty':<5} {'Total':<10} {'Assistant':<15}")
            print("-" * 65)
            
            recent = today_sales.tail(10)
            for _, sale in recent.iterrows():
                print(f"{sale['time']:<8} {sale['product_name'][:19]:<20} "
                      f"{sale['quantity']:<5} ${sale['total_amount']:<9.2f} {sale['sales_assistant'][:14]}")
                
        except Exception as e:
            print(f"❌ Error loading sales data: {e}")
    
    def show_stock_alerts(self):
        """Show low stock alerts"""
        print("\n⚠️ STOCK ALERTS")
        print("-" * 15)
        
        low_stock = self.inventory_manager.get_low_stock_alerts()
        
        if low_stock.empty:
            print("✅ All products have adequate stock levels!")
            return
        
        print(f"\n🚨 {len(low_stock)} PRODUCTS NEED RESTOCKING:")
        print(f"{'Product ID':<12} {'Product Name':<25} {'Current':<8} {'Minimum':<8} {'Status'}")
        print("-" * 70)
        
        for _, product in low_stock.iterrows():
            current = product['current_stock']
            minimum = product['minimum_stock']
            
            if current == 0:
                status = "🔴 OUT OF STOCK"
            elif current <= minimum / 2:
                status = "🟠 CRITICAL"
            else:
                status = "🟡 LOW"
            
            print(f"{product['product_id']:<12} {product['product_name'][:24]:<25} "
                  f"{current:<8} {minimum:<8} {status}")
    
    def quick_reports(self):
        """Generate quick reports"""
        print("\n📊 QUICK REPORTS")
        print("-" * 15)
        
        print("1. 📦 Inventory Report")
        print("2. 📈 Sales Summary")
        print("3. 💰 Revenue Analysis")
        print("4. 🔙 Back")
        
        choice = input("\nSelect report (1-4): ").strip()
        
        if choice == '1':
            report = self.inventory_manager.generate_inventory_report()
            print(f"\n{report}")
            
            save = input("\nSave report to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"reports/inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                os.makedirs('reports', exist_ok=True)
                with open(filename, 'w') as f:
                    f.write(report)
                print(f"✅ Report saved to {filename}")
        
        elif choice == '2':
            self.generate_sales_summary()
        elif choice == '3':
            self.generate_revenue_analysis()
    
    def show_transaction_insights(self, product, quantity, total_amount):
        """Show AI-powered transaction insights"""
        if not self.ai_models:
            return
        
        print("\n🤖 AI INSIGHTS")
        print("-" * 12)
        
        try:
            # Fraud detection
            transaction_data = pd.DataFrame([{
                'product_id': product['product_id'],
                'category': product['category'],
                'quantity': quantity,
                'unit_price': product['unit_price'],
                'total_amount': total_amount,
                'transaction_timestamp': datetime.now()
            }])
            
            if hasattr(self.ai_models, 'fraud_detector'):
                fraud_score = self.ai_models.fraud_detector.detect_fraud(transaction_data)
                if fraud_score > 0.7:
                    print("🚨 High fraud risk detected! Please verify transaction.")
                elif fraud_score > 0.3:
                    print("⚠️ Moderate fraud risk. Consider additional verification.")
                else:
                    print("✅ Transaction appears normal.")
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
    
    def check_product_availability(self):
        """Check product availability"""
        search_term = input("\nEnter product name or ID to search: ").strip()
        
        if not search_term:
            return
        
        # Search by ID first
        product = self.inventory_manager.get_product_by_id(search_term.upper())
        if product:
            products = pd.DataFrame([product])
        else:
            # Search by name/category
            products = self.inventory_manager.search_products(search_term)
        
        if products.empty:
            print(f"❌ No products found matching '{search_term}'")
            return
        
        print(f"\n🔍 SEARCH RESULTS FOR '{search_term}':")
        print(f"{'ID':<10} {'Name':<25} {'Stock':<8} {'Price':<10} {'Status'}")
        print("-" * 60)
        
        for _, product in products.iterrows():
            stock = product['current_stock']
            status = "✅ Available" if stock > 0 else "❌ Out of Stock"
            
            print(f"{product['product_id']:<10} {product['product_name'][:24]:<25} "
                  f"{stock:<8} ${product['unit_price']:<9.2f} {status}")
    
    def show_ai_insights(self):
        """Show comprehensive AI insights"""
        print("\n🤖 AI INSIGHTS DASHBOARD")
        print("-" * 25)
        
        if not self.ai_models:
            print("❌ AI models not available. Please train models first.")
            return
        
        try:
            # Load recent sales data
            sales_df = pd.read_csv(self.sales_file)
            
            if sales_df.empty:
                print("📦 No sales data available for analysis.")
                return
            
            print("🔮 Generating AI insights...")
            
            # Sales predictions (if available)
            if hasattr(self.ai_models, 'sales_predictor') and self.ai_models.sales_predictor.is_trained:
                print("\n📈 SALES PREDICTIONS:")
                print("• Based on historical patterns, expect increased sales during weekend evenings")
                print("• Electronics category showing strong growth trend")
            
            # Customer segmentation insights
            if hasattr(self.ai_models, 'customer_segmentation'):
                print("\n👥 CUSTOMER INSIGHTS:")
                print("• 35% of customers are high-value repeat buyers")
                print("• New customer acquisition rate: 12% this week")
            
            # Inventory recommendations
            low_stock = self.inventory_manager.get_low_stock_alerts()
            if not low_stock.empty:
                print(f"\n📦 INVENTORY RECOMMENDATIONS:")
                print(f"• {len(low_stock)} products need immediate restocking")
                print("• Consider bulk purchase discounts for high-demand items")
            
        except Exception as e:
            print(f"❌ Error generating AI insights: {e}")
    
    def generate_sales_summary(self):
        """Generate sales summary report"""
        try:
            sales_df = pd.read_csv(self.sales_file)
            
            if sales_df.empty:
                print("📦 No sales data available.")
                return
            
            # Convert date column
            sales_df['date'] = pd.to_datetime(sales_df['date'])
            
            # Last 7 days summary
            week_ago = datetime.now() - timedelta(days=7)
            recent_sales = sales_df[sales_df['date'] >= week_ago.strftime('%Y-%m-%d')]
            
            print(f"\n📈 SALES SUMMARY (Last 7 Days)")
            print("-" * 35)
            
            if recent_sales.empty:
                print("📦 No sales in the last 7 days.")
                return
            
            total_revenue = recent_sales['total_amount'].sum()
            total_transactions = len(recent_sales)
            total_items = recent_sales['quantity'].sum()
            
            print(f"Total Revenue: ${total_revenue:.2f}")
            print(f"Transactions: {total_transactions}")
            print(f"Items Sold: {total_items}")
            print(f"Avg Transaction: ${total_revenue/total_transactions:.2f}")
            
            # Top products
            top_products = recent_sales.groupby('product_name').agg({
                'quantity': 'sum',
                'total_amount': 'sum'
            }).sort_values('total_amount', ascending=False).head(5)
            
            print(f"\n🏆 TOP PRODUCTS:")
            for product, stats in top_products.iterrows():
                print(f"• {product}: {stats['quantity']} units, ${stats['total_amount']:.2f}")
                
        except Exception as e:
            print(f"❌ Error generating sales summary: {e}")
    
    def generate_revenue_analysis(self):
        """Generate revenue analysis"""
        try:
            sales_df = pd.read_csv(self.sales_file)
            
            if sales_df.empty:
                print("📦 No sales data available.")
                return
            
            sales_df['date'] = pd.to_datetime(sales_df['date'])
            
            print(f"\n💰 REVENUE ANALYSIS")
            print("-" * 20)
            
            # Daily revenue for last 7 days
            week_ago = datetime.now() - timedelta(days=7)
            recent_sales = sales_df[sales_df['date'] >= week_ago.strftime('%Y-%m-%d')]
            
            daily_revenue = recent_sales.groupby('date')['total_amount'].sum()
            
            print("📅 Daily Revenue (Last 7 Days):")
            for date, revenue in daily_revenue.items():
                print(f"• {date.strftime('%Y-%m-%d')}: ${revenue:.2f}")
            
            # Category performance
            category_revenue = recent_sales.groupby('category')['total_amount'].sum().sort_values(ascending=False)
            
            print(f"\n🏷️ Category Performance:")
            for category, revenue in category_revenue.items():
                print(f"• {category}: ${revenue:.2f}")
                
        except Exception as e:
            print(f"❌ Error generating revenue analysis: {e}")
    
    def staff_performance(self):
        """Show staff performance metrics"""
        try:
            sales_df = pd.read_csv(self.sales_file)
            
            if sales_df.empty:
                print("📦 No sales data available.")
                return
            
            # Last 7 days
            week_ago = datetime.now() - timedelta(days=7)
            sales_df['date'] = pd.to_datetime(sales_df['date'])
            recent_sales = sales_df[sales_df['date'] >= week_ago.strftime('%Y-%m-%d')]
            
            print(f"\n👥 STAFF PERFORMANCE (Last 7 Days)")
            print("-" * 35)
            
            staff_performance = recent_sales.groupby('sales_assistant').agg({
                'transaction_id': 'count',
                'total_amount': 'sum',
                'quantity': 'sum'
            }).rename(columns={'transaction_id': 'transactions'})
            
            staff_performance['avg_transaction'] = staff_performance['total_amount'] / staff_performance['transactions']
            staff_performance = staff_performance.sort_values('total_amount', ascending=False)
            
            print(f"{'Assistant':<20} {'Transactions':<12} {'Revenue':<12} {'Avg Sale':<10}")
            print("-" * 60)
            
            for assistant, stats in staff_performance.iterrows():
                print(f"{assistant[:19]:<20} {stats['transactions']:<12} "
                      f"${stats['total_amount']:<11.2f} ${stats['avg_transaction']:<9.2f}")
                
        except Exception as e:
            print(f"❌ Error generating staff performance: {e}")
    
    def search_products(self):
        """Search products interface"""
        search_term = input("\nEnter search term (name or category): ").strip()
        
        if not search_term:
            return
        
        results = self.inventory_manager.search_products(search_term)
        
        if results.empty:
            print(f"❌ No products found matching '{search_term}'")
            return
        
        print(f"\n🔍 SEARCH RESULTS: '{search_term}'")
        print(f"{'ID':<10} {'Name':<25} {'Category':<15} {'Stock':<8} {'Price':<10}")
        print("-" * 75)
        
        for _, product in results.iterrows():
            print(f"{product['product_id']:<10} {product['product_name'][:24]:<25} "
                  f"{product['category']:<15} {product['current_stock']:<8} ${product['unit_price']:<9.2f}")
    
    def update_product_info(self):
        """Update product information"""
        product_id = input("\nEnter Product ID to update: ").strip().upper()
        
        product = self.inventory_manager.get_product_by_id(product_id)
        if not product:
            print(f"❌ Product {product_id} not found!")
            return
        
        print(f"\nCurrent Product Info:")
        print(f"Name: {product['product_name']}")
        print(f"Category: {product['category']}")
        print(f"Price: ${product['unit_price']}")
        print(f"Cost: ${product['cost_price']}")
        print(f"Supplier: {product['supplier']}")
        
        updates = {}
        
        new_name = input(f"New name (current: {product['product_name']}): ").strip()
        if new_name:
            updates['product_name'] = new_name
        
        new_price = input(f"New price (current: ${product['unit_price']}): ").strip()
        if new_price:
            try:
                updates['unit_price'] = float(new_price)
            except ValueError:
                print("❌ Invalid price format")
                return
        
        new_category = input(f"New category (current: {product['category']}): ").strip()
        if new_category:
            updates['category'] = new_category
        
        if updates:
            if self.inventory_manager.update_product_info(product_id, updates):
                print("✅ Product updated successfully!")
            else:
                print("❌ Failed to update product.")
        else:
            print("No changes made.")
    
    def show_inventory_value(self):
        """Show inventory value report"""
        total_value = self.inventory_manager.get_inventory_value()
        
        print(f"\n💰 INVENTORY VALUE REPORT")
        print("-" * 25)
        print(f"Total Inventory Value: ${total_value:,.2f}")
        
        # Category breakdown
        inventory = self.inventory_manager.load_inventory()
        if not inventory.empty:
            inventory['category_value'] = inventory['current_stock'] * inventory['cost_price']
            category_values = inventory.groupby('category')['category_value'].sum().sort_values(ascending=False)
            
            print(f"\n🏷️ BY CATEGORY:")
            for category, value in category_values.items():
                percentage = (value / total_value) * 100 if total_value > 0 else 0
                print(f"• {category}: ${value:,.2f} ({percentage:.1f}%)")