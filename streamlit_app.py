"""
Horizon Enterprise AI-Powered POS System - Streamlit Web Interface
================================================================

A comprehensive web-based POS system with AI-powered features for:
- Sales Management
- Inventory Management  
- Customer Insights
- Fraud Detection
- Real-time Analytics

Author: [Your Name]
Course: AI for Software Engineering
Date: November 8, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import os
import sys
import json
import time

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Currency formatting function for Lesotho Maloti (LSL)
def format_currency(amount):
    """Format amount in Lesotho Maloti (LSL) currency"""
    return f"M {amount:,.2f}"

# Configure Streamlit page
st.set_page_config(
    page_title="Horizon AI POS System",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all system data"""
    try:
        # Load inventory
        if os.path.exists('data/inventory.csv'):
            inventory = pd.read_csv('data/inventory.csv')
        else:
            inventory = pd.DataFrame(columns=['product_id', 'product_name', 'category', 'unit_price', 'stock_quantity', 'reorder_level'])
        
        # Load transactions
        if os.path.exists('data/transactions.csv'):
            transactions = pd.read_csv('data/transactions.csv')
            transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        else:
            transactions = pd.DataFrame()
        
        # Load customers
        if os.path.exists('data/customers.csv'):
            customers = pd.read_csv('data/customers.csv')
        else:
            customers = pd.DataFrame()
            
        return inventory, transactions, customers
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def save_inventory(inventory_df):
    """Save inventory data"""
    try:
        os.makedirs('data', exist_ok=True)
        inventory_df.to_csv('data/inventory.csv', index=False)
        return True
    except Exception as e:
        st.error(f"Error saving inventory: {e}")
        return False

def save_transaction(transaction_data):
    """Save a new transaction"""
    try:
        os.makedirs('data', exist_ok=True)
        
        # Load existing transactions or create new DataFrame
        if os.path.exists('data/transactions.csv'):
            transactions = pd.read_csv('data/transactions.csv')
        else:
            transactions = pd.DataFrame()
        
        # Add new transaction
        new_transaction = pd.DataFrame([transaction_data])
        transactions = pd.concat([transactions, new_transaction], ignore_index=True)
        
        # Save back to CSV
        transactions.to_csv('data/transactions.csv', index=False)
        return True
    except Exception as e:
        st.error(f"Error saving transaction: {e}")
        return False

def load_ai_models():
    """Load AI models if available"""
    try:
        from src.ai_models.sales_predictor import SalesPredictor
        from src.ai_models.customer_segmentation import CustomerSegmentation
        from src.ai_models.fraud_detector_fixed import FraudDetectorFixed
        
        models = {}
        
        # Load Sales Predictor
        if os.path.exists('models/sales_predictor.pkl'):
            sales_predictor = SalesPredictor()
            sales_predictor.load_model('models/sales_predictor.pkl')
            models['sales_predictor'] = sales_predictor
        
        # Load Customer Segmentation
        if os.path.exists('models/customer_segmentation.pkl'):
            customer_segmentation = CustomerSegmentation()
            customer_segmentation.load_model('models/customer_segmentation.pkl')
            models['customer_segmentation'] = customer_segmentation
        
        # Load Fraud Detector
        if os.path.exists('models/fraud_detector.pkl'):
            fraud_detector = FraudDetectorFixed()
            fraud_detector.load_model('models/fraud_detector.pkl')
            models['fraud_detector'] = fraud_detector
            
        return models
    except Exception as e:
        st.warning(f"AI models not available: {e}")
        return {}

def main_dashboard():
    """Main dashboard with key metrics"""
    st.markdown('<h1 class="main-header">🏪 Horizon AI POS Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    inventory, transactions, customers = load_data()
    
    # Key Metrics Row - Real Data Display
    if not transactions.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = transactions['total_amount'].sum()
            st.metric("💰 Total Sales", format_currency(total_sales))
            
        with col2:
            total_transactions = len(transactions) 
            st.metric("🧾 Total Transactions", f"{total_transactions:,}")
            
        with col3:
            avg_transaction = transactions['total_amount'].mean()
            st.metric("📊 Avg Transaction", format_currency(avg_transaction))
            
        with col4:
            unique_customers = transactions['customer_id'].nunique()
            st.metric("👥 Unique Customers", f"{unique_customers:,}")
    else:
        # Show placeholder metrics when no transaction data exists
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Total Sales", format_currency(0))
        with col2:
            st.metric("🧾 Total Transactions", "0")
        with col3:
            st.metric("📊 Avg Transaction", format_currency(0))
        with col4:
            st.metric("👥 Unique Customers", "0")
    
    # Enhanced Analytics Charts - Real Data Visualization
    if not transactions.empty:
        st.subheader("📊 Real-Time Sales Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Daily Sales Trend")
            daily_sales = transactions.groupby('transaction_date')['total_amount'].sum().reset_index()
            daily_sales['transaction_date'] = pd.to_datetime(daily_sales['transaction_date'])
            
            fig = px.line(
                daily_sales, 
                x='transaction_date', 
                y='total_amount',
                title=f"Daily Sales Revenue (Total: {format_currency(daily_sales['total_amount'].sum())})",
                labels={'total_amount': 'Sales Amount (LSL)', 'transaction_date': 'Date'}
            )
            fig.update_traces(line_color='#1f77b4', line_width=3)
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏷️ Sales by Category")
            category_sales = transactions.groupby('category')['total_amount'].sum().reset_index()
            category_sales = category_sales.sort_values('total_amount', ascending=False)
            
            fig = px.pie(
                category_sales, 
                values='total_amount', 
                names='category',
                title=f"Revenue Distribution ({len(category_sales)} Categories)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
        # Additional Analytics Row
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("💳 Payment Methods")
            payment_stats = transactions['payment_method'].value_counts().reset_index()
            payment_stats.columns = ['Payment Method', 'Transactions']
            
            fig = px.bar(
                payment_stats,
                x='Payment Method',
                y='Transactions',
                title="Transaction Count by Payment Method",
                color='Transactions',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col4:
            st.subheader("🏆 Top Products")
            product_sales = transactions.groupby('product_name')['total_amount'].sum().reset_index()
            product_sales = product_sales.sort_values('total_amount', ascending=False).head(10)
            
            fig = px.bar(
                product_sales,
                x='total_amount',
                y='product_name',
                orientation='h',
                title="Top 10 Products by Revenue",
                labels={'total_amount': 'Revenue (LSL)', 'product_name': 'Product'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No transaction data available for analytics. Complete some sales to see charts!")
    
    # Enhanced Inventory Status - Real Stock Data
    st.subheader("📦 Real-Time Inventory Status")
    
    if not inventory.empty:
        # Inventory overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_products = len(inventory)
            st.metric("📦 Total Products", f"{total_products:,}")
            
        with col2:
            total_stock_units = inventory['stock_quantity'].sum()
            st.metric("📊 Total Stock Units", f"{total_stock_units:,}")
            
        with col3:
            total_stock_value = (inventory['unit_price'] * inventory['stock_quantity']).sum()
            st.metric("💎 Total Stock Value", format_currency(total_stock_value))
            
        with col4:
            out_of_stock = len(inventory[inventory['stock_quantity'] == 0])
            st.metric("🚫 Out of Stock", f"{out_of_stock:,}")
        
        # Low stock alerts
        low_stock = inventory[inventory['stock_quantity'] <= inventory['reorder_level']]
        if not low_stock.empty:
            st.warning(f"⚠️ {len(low_stock)} products are low in stock!")
            st.dataframe(
                low_stock[['product_name', 'stock_quantity', 'reorder_level', 'unit_price']].style.format({
                    'unit_price': lambda x: format_currency(x)
                }),
                use_container_width=True
            )
        else:
            st.success("✅ All products are adequately stocked!")
            
        # Current inventory overview table
        st.subheader("� Current Inventory Overview")
        inventory_display = inventory.copy()
        inventory_display['Stock Value'] = inventory_display['unit_price'] * inventory_display['stock_quantity']
        
        # Format the display
        display_df = inventory_display[['product_name', 'category', 'stock_quantity', 'unit_price', 'Stock Value', 'reorder_level']].copy()
        display_df.columns = ['Product Name', 'Category', 'Stock Qty', 'Unit Price', 'Stock Value', 'Reorder Level']
        
        st.dataframe(
            display_df.style.format({
                'Unit Price': lambda x: format_currency(x),
                'Stock Value': lambda x: format_currency(x)
            }),
            use_container_width=True
        )
    else:
        st.warning("📦 No inventory data available. Please add products through the Manager Interface.")
        # Show placeholder metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Total Products", "0")
        with col2:
            st.metric("� Total Stock Units", "0")
        with col3:
            st.metric("💎 Total Stock Value", format_currency(0))
        with col4:
            st.metric("🚫 Out of Stock", "0")

def sales_interface():
    """Sales assistant interface"""
    st.markdown('<h1 class="main-header">🛒 Sales Interface</h1>', unsafe_allow_html=True)
    
    # Load data
    inventory, transactions, customers = load_data()
    
    if inventory.empty:
        st.warning("No inventory data available. Please add products first in the Manager Interface.")
        return
    
    # Quick Sale Mode Toggle
    st.subheader("⚡ Sales Mode")
    sale_mode = st.radio("Choose Transaction Mode:", 
                        ["🛒 Regular Sale", "⚡ Quick Sale (Cash Only)"], 
                        horizontal=True)
    
    if sale_mode == "⚡ Quick Sale (Cash Only)":
        # Quick Sale Interface
        st.markdown("### ⚡ Quick Cash Sale")
        st.info("Perfect for busy periods - optimized for speed!")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            available_products = inventory[inventory['stock_quantity'] > 0]
            product_options = available_products['product_name'].tolist()
            quick_product = st.selectbox("🔍 Product", product_options, key="quick_product")
        
        with col2:
            if quick_product:
                product_data = available_products[available_products['product_name'] == quick_product].iloc[0]
                quick_qty = st.number_input("Qty", min_value=1, max_value=int(product_data['stock_quantity']), value=1, key="quick_qty")
        
        with col3:
            if quick_product:
                quick_total = product_data['unit_price'] * quick_qty
                quick_cash = st.number_input(f"Cash Received (Total: {format_currency(quick_total)})", 
                                           min_value=0.0, value=float(quick_total), step=0.01, key="quick_cash")
                quick_change = quick_cash - quick_total
        
        # Quick process button
        if quick_product and st.button("⚡ QUICK PROCESS", type="primary", key="quick_process"):
            if quick_cash >= quick_total:
                # Quick transaction
                quick_transaction = {
                    'transaction_id': f"QTXN_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'customer_id': f"WALK_IN_{datetime.now().strftime('%H%M%S')}",
                    'customer_name': "Walk-in Customer",
                    'product_id': product_data['product_id'],
                    'product_name': quick_product,
                    'category': product_data['category'],
                    'quantity': quick_qty,
                    'unit_price': product_data['unit_price'],
                    'subtotal': quick_total,
                    'discount_amount': 0,
                    'total_amount': quick_total,
                    'payment_method': 'Cash',
                    'payment_received': quick_cash,
                    'change_due': quick_change,
                    'transaction_date': datetime.now().strftime('%Y-%m-%d'),
                    'transaction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'cashier': 'Quick Sale'
                }
                
                # Update inventory
                inventory.loc[inventory['product_name'] == quick_product, 'stock_quantity'] -= quick_qty
                
                # Save transaction
                if save_transaction(quick_transaction) and save_inventory(inventory):
                    st.success("✅ Quick sale completed!")
                    if quick_change > 0:
                        st.success(f"💰 **CHANGE: {format_currency(quick_change)}**")
                    st.cache_data.clear()
            else:
                st.error(f"❌ Insufficient cash! Need {format_currency(quick_total - quick_cash)} more")
        
        st.markdown("---")
        
    # Regular Sales form
    with st.form("sales_form"):
        st.subheader("New Sale Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer information
            customer_id = st.text_input("Customer ID", value=f"CUST_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            customer_name = st.text_input("Customer Name")
            
            # Product selection
            available_products = inventory[inventory['stock_quantity'] > 0]
            if available_products.empty:
                st.error("No products available in stock!")
                return
            
            product_options = available_products['product_name'].tolist()
            selected_product = st.selectbox("Select Product", product_options)
            
            if selected_product:
                product_data = available_products[available_products['product_name'] == selected_product].iloc[0]
                st.info(f"Price: {format_currency(product_data['unit_price'])} | Stock: {product_data['stock_quantity']} units")
        
        with col2:
            # Transaction details
            if selected_product:
                quantity = st.number_input("Quantity", min_value=1, max_value=int(product_data['stock_quantity']), value=1)
                
                # Calculate subtotal
                subtotal = product_data['unit_price'] * quantity
                
                # Discount options
                discount_type = st.radio("Discount Type", ["Percentage", "Fixed Amount", "None"])
                discount_amount = 0
                
                if discount_type == "Percentage":
                    discount_percent = st.number_input("Discount %", min_value=0.0, max_value=50.0, value=0.0)
                    discount_amount = subtotal * (discount_percent / 100)
                elif discount_type == "Fixed Amount":
                    discount_amount = st.number_input("Discount Amount (M)", min_value=0.0, max_value=float(subtotal), value=0.0)
                
                # Calculate total after discount
                total_due = subtotal - discount_amount
                
                # Payment method
                payment_method = st.selectbox("Payment Method", ["Cash", "Credit Card", "Debit Card", "Mobile Payment"])
                
                # Payment amount (only for cash)
                payment_received = 0
                change_due = 0
                
                if payment_method == "Cash":
                    payment_received = st.number_input(
                        f"Cash Received (M)", 
                        min_value=0.0, 
                        value=float(total_due),
                        step=0.01,
                        help="Enter the amount of cash received from customer"
                    )
                    change_due = payment_received - total_due
                    
                    if payment_received < total_due:
                        st.error(f"❌ Insufficient payment! Need {format_currency(total_due - payment_received)} more")
                    elif change_due > 0:
                        st.success(f"💰 Change due: {format_currency(change_due)}")
                    else:
                        st.info("✅ Exact payment received")
                else:
                    payment_received = total_due  # For card payments, assume exact amount
                    st.info("💳 Electronic payment - no change required")
                
                # Order Summary
                st.markdown("---")
                st.subheader("📋 Order Summary")
                st.markdown(f"""
                **Product**: {selected_product}  
                **Unit Price**: {format_currency(product_data['unit_price'])}  
                **Quantity**: {quantity}  
                **Subtotal**: {format_currency(subtotal)}  
                **Discount**: {format_currency(discount_amount)}  
                **Total Due**: {format_currency(total_due)}  
                **Payment Method**: {payment_method}  
                **Amount Received**: {format_currency(payment_received)}  
                **Change**: {format_currency(change_due)}
                """)
                
                # Store values for transaction processing
                globals()['current_transaction'] = {
                    'subtotal': subtotal,
                    'discount_amount': discount_amount,
                    'total_amount': total_due,
                    'payment_received': payment_received,
                    'change_due': change_due
                }
        
        # Submit transaction
        submitted = st.form_submit_button("🧾 Process Sale", type="primary", 
                                         disabled=selected_product is None or 
                                         (payment_method == "Cash" and payment_received < total_due if 'total_due' in locals() and 'payment_received' in locals() else False))
        
        if submitted and selected_product:
            # Get transaction data from globals (stored above)
            trans_data = globals().get('current_transaction', {})
            
            # Validate payment for cash transactions
            if payment_method == "Cash" and trans_data.get('change_due', 0) < 0:
                st.error("❌ Cannot process sale - insufficient payment received!")
                return
            
            # Create transaction record
            transaction_data = {
                'transaction_id': f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'customer_id': customer_id,
                'customer_name': customer_name,
                'product_id': product_data['product_id'],
                'product_name': selected_product,
                'category': product_data['category'],
                'quantity': quantity,
                'unit_price': product_data['unit_price'],
                'subtotal': trans_data.get('subtotal', 0),
                'discount_amount': trans_data.get('discount_amount', 0),
                'total_amount': trans_data.get('total_amount', 0),
                'payment_method': payment_method,
                'payment_received': trans_data.get('payment_received', 0),
                'change_due': trans_data.get('change_due', 0),
                'transaction_date': datetime.now().strftime('%Y-%m-%d'),
                'transaction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cashier': st.session_state.get('user_name', 'Sales Assistant')
            }
            
            # Update inventory
            inventory.loc[inventory['product_name'] == selected_product, 'stock_quantity'] -= quantity
            
            # Save transaction and inventory
            if save_transaction(transaction_data) and save_inventory(inventory):
                st.success("✅ Transaction completed successfully!")
                st.balloons()
                
                # Show receipt
                st.markdown("---")
                st.subheader("🧾 Receipt")
                
                # Highlight change due if cash payment
                if payment_method == "Cash" and transaction_data['change_due'] > 0:
                    st.success(f"💰 **CHANGE DUE: {format_currency(transaction_data['change_due'])}**")
                
                receipt_content = f"""
                **🏪 HORIZON ENTERPRISE**  
                📍 Your Trusted POS System  
                ═══════════════════════════════════
                
                **Transaction Details:**  
                🆔 Transaction ID: {transaction_data['transaction_id']}  
                📅 Date: {transaction_data['transaction_timestamp']}  
                👤 Customer: {customer_name} ({customer_id})  
                👨‍💼 Cashier: {transaction_data['cashier']}
                
                ═══════════════════════════════════
                **ITEMS PURCHASED:**
                📦 {selected_product}  
                    💰 {format_currency(transaction_data['unit_price'])} × {quantity} = {format_currency(transaction_data['subtotal'])}
                
                ───────────────────────────────────
                💵 Subtotal: {format_currency(transaction_data['subtotal'])}  
                🏷️ Discount: -{format_currency(transaction_data['discount_amount'])}  
                **💳 TOTAL DUE: {format_currency(transaction_data['total_amount'])}**
                
                ═══════════════════════════════════
                **PAYMENT DETAILS:**  
                💼 Method: {payment_method}  
                💵 Received: {format_currency(transaction_data['payment_received'])}  
                💰 Change: {format_currency(transaction_data['change_due'])}
                
                ═══════════════════════════════════
                ✨ Thank you for shopping with us! ✨  
                🔄 Please keep this receipt for returns  
                📞 Support: horizon-support@email.com
                """
                
                st.markdown(receipt_content)
                
                # Cash handling instructions for sales assistant
                if payment_method == "Cash":
                    if transaction_data['change_due'] > 0:
                        st.info(f"💡 **Sales Assistant:** Give {format_currency(transaction_data['change_due'])} change to customer")
                        
                        # Suggest change breakdown for large amounts
                        if transaction_data['change_due'] >= 20:
                            change = transaction_data['change_due']
                            bills_20 = int(change // 20)
                            change %= 20
                            bills_10 = int(change // 10)
                            change %= 10
                            bills_5 = int(change // 5)
                            change %= 5
                            bills_1 = int(change // 1)
                            coins = change % 1
                            
                            breakdown = "💵 **Suggested Change Breakdown:**\n"
                            if bills_20 > 0: breakdown += f"• M 20 bills: {bills_20}\n"
                            if bills_10 > 0: breakdown += f"• M 10 bills: {bills_10}\n" 
                            if bills_5 > 0: breakdown += f"• M 5 bills: {bills_5}\n"
                            if bills_1 > 0: breakdown += f"• M 1 bills: {bills_1}\n"
                            if coins > 0: breakdown += f"• Coins: {format_currency(coins)}\n"
                            
                            st.info(breakdown)
                    else:
                        st.success("✅ **Exact payment received - no change needed**")
                
                # Clear cache to refresh data
                st.cache_data.clear()
            else:
                st.error("❌ Error processing transaction!")

def manager_interface():
    """Manager interface for inventory and analytics"""
    st.markdown('<h1 class="main-header">👔 Manager Interface</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📦 Inventory Management", "📊 Sales Analytics", "👥 Customer Insights", "🤖 AI Insights"])
    
    with tab1:
        inventory_management()
    
    with tab2:
        sales_analytics()
    
    with tab3:
        customer_insights()
    
    with tab4:
        ai_insights()

def inventory_management():
    """Inventory management interface"""
    st.subheader("📦 Inventory Management")
    
    # Load inventory
    inventory, _, _ = load_data()
    
    # Inventory Status Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_products = len(inventory) if not inventory.empty else 0
        st.metric("📦 Total Products", total_products)
    with col2:
        total_stock = inventory['stock_quantity'].sum() if not inventory.empty else 0
        st.metric("📊 Total Stock Units", f"{total_stock:,}")
    with col3:
        total_value = (inventory['unit_price'] * inventory['stock_quantity']).sum() if not inventory.empty else 0
        st.metric("💰 Inventory Value", format_currency(total_value))
    with col4:
        low_stock_count = len(inventory[inventory['stock_quantity'] <= inventory['reorder_level']]) if not inventory.empty else 0
        st.metric("⚠️ Low Stock Items", low_stock_count)
    
    # Inventory Actions
    st.markdown("---")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔄 Reset Inventory", help="Clear all inventory and start fresh"):
            if st.session_state.get('confirm_reset', False):
                # Create empty inventory
                empty_inventory = pd.DataFrame(columns=['product_id', 'product_name', 'category', 'unit_price', 'stock_quantity', 'reorder_level', 'description', 'date_added'])
                if save_inventory(empty_inventory):
                    st.success("✅ Inventory reset successfully!")
                    st.session_state['confirm_reset'] = False
                    st.cache_data.clear()
                else:
                    st.error("❌ Error resetting inventory!")
            else:
                st.session_state['confirm_reset'] = True
                st.warning("⚠️ Click again to confirm inventory reset!")
    
    with action_col2:
        if not inventory.empty:
            csv_data = inventory.to_csv(index=False)
            st.download_button(
                label="💾 Backup Inventory",
                data=csv_data,
                file_name=f"inventory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download current inventory as backup"
            )
    
    with action_col3:
        if st.button("📋 Sample Template", help="Download a sample CSV template"):
            sample_data = pd.DataFrame({
                'product_name': ['Sample Product 1', 'Sample Product 2'],
                'category': ['Electronics', 'Clothing'],
                'unit_price': [99.99, 29.99],
                'stock_quantity': [50, 100],
                'reorder_level': [10, 20],
                'description': ['Sample electronics item', 'Sample clothing item']
            })
            
            sample_csv = sample_data.to_csv(index=False)
            st.download_button(
                label="📋 Download Template",
                data=sample_csv,
                file_name="inventory_template.csv",
                mime="text/csv"
            )
    
    # Add new product
    with st.expander("➕ Add New Product"):
        with st.form("add_product_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                product_name = st.text_input("Product Name")
                category = st.selectbox("Category", ["Electronics", "Clothing", "Food", "Books", "Home", "Sports", "Other"])
                unit_price = st.number_input("Unit Price (M)", min_value=0.01, value=10.00)
            
            with col2:
                stock_quantity = st.number_input("Initial Stock Quantity", min_value=0, value=100)
                reorder_level = st.number_input("Reorder Level", min_value=0, value=20)
                product_description = st.text_area("Product Description")
            
            if st.form_submit_button("Add Product"):
                if product_name:
                    new_product = {
                        'product_id': f"PROD_{len(inventory) + 1:04d}",
                        'product_name': product_name,
                        'category': category,
                        'unit_price': unit_price,
                        'stock_quantity': stock_quantity,
                        'reorder_level': reorder_level,
                        'description': product_description,
                        'date_added': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    # Add to inventory
                    inventory = pd.concat([inventory, pd.DataFrame([new_product])], ignore_index=True)
                    
                    if save_inventory(inventory):
                        st.success(f"✅ Product '{product_name}' added successfully!")
                        st.cache_data.clear()
                    else:
                        st.error("❌ Error adding product!")
                else:
                    st.error("Product name is required!")
    
    # Inventory Management Options
    st.subheader("📋 Inventory Management Options")
    
    tab1, tab2, tab3 = st.tabs(["📊 Current Inventory", "✏️ Edit Products", "🗑️ Remove Products"])
    
    with tab1:
        # Current inventory display
        if not inventory.empty:
            st.subheader("Current Inventory")
            
            # Search and filter
            col1, col2, col3 = st.columns(3)
            with col1:
                search_term = st.text_input("🔍 Search Products", key="search_current")
            with col2:
                category_filter = st.selectbox("Filter by Category", ["All"] + list(inventory['category'].unique()), key="filter_current")
            with col3:
                stock_filter = st.selectbox("Stock Status", ["All", "In Stock", "Low Stock", "Out of Stock"], key="status_current")
            
            # Apply filters
            filtered_inventory = inventory.copy()
            
            if search_term:
                filtered_inventory = filtered_inventory[
                    filtered_inventory['product_name'].str.contains(search_term, case=False, na=False)
                ]
            
            if category_filter != "All":
                filtered_inventory = filtered_inventory[filtered_inventory['category'] == category_filter]
            
            if stock_filter == "In Stock":
                filtered_inventory = filtered_inventory[filtered_inventory['stock_quantity'] > filtered_inventory['reorder_level']]
            elif stock_filter == "Low Stock":
                filtered_inventory = filtered_inventory[
                    (filtered_inventory['stock_quantity'] <= filtered_inventory['reorder_level']) & 
                    (filtered_inventory['stock_quantity'] > 0)
                ]
            elif stock_filter == "Out of Stock":
                filtered_inventory = filtered_inventory[filtered_inventory['stock_quantity'] == 0]
            
            # Display inventory with formatted prices
            display_inventory = filtered_inventory[['product_name', 'category', 'unit_price', 'stock_quantity', 'reorder_level']].copy()
            display_inventory['unit_price'] = display_inventory['unit_price'].apply(format_currency)
            st.dataframe(
                display_inventory,
                width='stretch',
                hide_index=True
            )
            
            # Quick stock update
            st.subheader("⚡ Quick Stock Update")
            col1, col2, col3 = st.columns(3)
            with col1:
                quick_product = st.selectbox("Select Product", filtered_inventory['product_name'].tolist(), key="quick_update")
            with col2:
                current_stock = filtered_inventory[filtered_inventory['product_name'] == quick_product]['stock_quantity'].iloc[0] if quick_product and not filtered_inventory.empty else 0
                st.info(f"Current Stock: {current_stock}")
                new_stock = st.number_input("New Stock Quantity", min_value=0, value=int(current_stock), key="new_stock")
            with col3:
                st.write("") # Spacer
                st.write("") # Spacer
                if st.button("🔄 Update Stock", key="update_quick"):
                    inventory.loc[inventory['product_name'] == quick_product, 'stock_quantity'] = new_stock
                    if save_inventory(inventory):
                        st.success(f"✅ Updated {quick_product} stock to {new_stock}")
                        st.cache_data.clear()
        else:
            st.info("No inventory items yet. Add your first product above!")
    
    with tab2:
        # Edit existing products
        st.subheader("✏️ Edit Product Details")
        
        if not inventory.empty:
            # Select product to edit
            edit_product = st.selectbox("Select Product to Edit", inventory['product_name'].tolist(), key="edit_select")
            
            if edit_product:
                # Get current product data
                current_product = inventory[inventory['product_name'] == edit_product].iloc[0]
                
                # Edit form
                with st.form("edit_product_form"):
                    st.subheader(f"Editing: {edit_product}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        edit_name = st.text_input("Product Name", value=current_product['product_name'])
                        edit_category = st.selectbox("Category", 
                                                   ["Electronics", "Clothing", "Food", "Books", "Home", "Sports", "Other"],
                                                   index=["Electronics", "Clothing", "Food", "Books", "Home", "Sports", "Other"].index(current_product['category']) if current_product['category'] in ["Electronics", "Clothing", "Food", "Books", "Home", "Sports", "Other"] else 0)
                        edit_price = st.number_input("Unit Price (M)", min_value=0.01, value=float(current_product['unit_price']))
                    
                    with col2:
                        edit_stock = st.number_input("Current Stock", min_value=0, value=int(current_product['stock_quantity']))
                        edit_reorder = st.number_input("Reorder Level", min_value=0, value=int(current_product['reorder_level']))
                        edit_description = st.text_area("Description", value=current_product.get('description', ''))
                    
                    if st.form_submit_button("💾 Save Changes", type="primary"):
                        # Update the product
                        inventory.loc[inventory['product_name'] == edit_product, 'product_name'] = edit_name
                        inventory.loc[inventory['product_name'] == edit_product, 'category'] = edit_category
                        inventory.loc[inventory['product_name'] == edit_product, 'unit_price'] = edit_price
                        inventory.loc[inventory['product_name'] == edit_product, 'stock_quantity'] = edit_stock
                        inventory.loc[inventory['product_name'] == edit_product, 'reorder_level'] = edit_reorder
                        inventory.loc[inventory['product_name'] == edit_product, 'description'] = edit_description
                        
                        if save_inventory(inventory):
                            st.success(f"✅ Product '{edit_name}' updated successfully!")
                            st.cache_data.clear()
                        else:
                            st.error("❌ Error updating product!")
        else:
            st.info("No products available to edit. Add some products first!")
    
    with tab3:
        # Remove products
        st.subheader("🗑️ Remove Products")
        
        if not inventory.empty:
            st.warning("⚠️ **Caution**: Removing products is permanent and cannot be undone!")
            
            # Select product to remove
            remove_product = st.selectbox("Select Product to Remove", inventory['product_name'].tolist(), key="remove_select")
            
            if remove_product:
                # Show product details
                product_to_remove = inventory[inventory['product_name'] == remove_product].iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **Product Details:**
                    - Name: {product_to_remove['product_name']}
                    - Category: {product_to_remove['category']}
                    - Price: {format_currency(product_to_remove['unit_price'])}
                    - Stock: {product_to_remove['stock_quantity']} units
                    """)
                
                with col2:
                    # Confirmation
                    confirm_text = st.text_input("Type 'DELETE' to confirm removal:", key="confirm_delete")
                    
                    if st.button("🗑️ Remove Product", type="secondary", key="remove_btn"):
                        if confirm_text.upper() == "DELETE":
                            # Remove the product
                            updated_inventory = inventory[inventory['product_name'] != remove_product]
                            
                            if save_inventory(updated_inventory):
                                st.success(f"✅ Product '{remove_product}' removed successfully!")
                                st.cache_data.clear()
                            else:
                                st.error("❌ Error removing product!")
                        else:
                            st.error("❌ Please type 'DELETE' to confirm removal!")
        else:
            st.info("No products available to remove.")
    
    # Bulk Import Section
    with st.expander("📤 Bulk Import/Export"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Import from CSV")
            uploaded_file = st.file_uploader("Upload inventory CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    new_inventory = pd.read_csv(uploaded_file)
                    
                    # Validate required columns
                    required_cols = ['product_name', 'category', 'unit_price', 'stock_quantity', 'reorder_level']
                    if all(col in new_inventory.columns for col in required_cols):
                        
                        st.write("Preview of imported data:")
                        st.dataframe(new_inventory.head(), width='stretch')
                        
                        if st.button("📥 Import Inventory"):
                            # Add product IDs and dates
                            new_inventory['product_id'] = [f"PROD_{i+len(inventory)+1:04d}" for i in range(len(new_inventory))]
                            new_inventory['date_added'] = datetime.now().strftime('%Y-%m-%d')
                            
                            # Combine with existing inventory
                            combined_inventory = pd.concat([inventory, new_inventory], ignore_index=True)
                            
                            if save_inventory(combined_inventory):
                                st.success(f"✅ Imported {len(new_inventory)} products successfully!")
                                st.cache_data.clear()
                            else:
                                st.error("❌ Error importing inventory!")
                    else:
                        st.error(f"❌ CSV must contain columns: {', '.join(required_cols)}")
                        
                except Exception as e:
                    st.error(f"❌ Error reading CSV file: {e}")
        
        with col2:
            st.subheader("📤 Export to CSV")
            
            if not inventory.empty:
                csv_data = inventory.to_csv(index=False)
                
                st.download_button(
                    label="📤 Download Inventory CSV",
                    data=csv_data,
                    file_name=f"horizon_inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.info(f"Ready to export {len(inventory)} products")
            else:
                st.info("No inventory data to export")

def sales_analytics():
    """Enhanced sales analytics interface with real product data"""
    st.subheader("📊 Sales Analytics")
    
    # Load data
    inventory, transactions, customers = load_data()
    
    if transactions.empty:
        st.warning("No transaction data available.")
        return
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=transactions['transaction_date'].min().date())
    with col2:
        end_date = st.date_input("End Date", value=transactions['transaction_date'].max().date())
    
    # Filter transactions
    filtered_transactions = transactions[
        (transactions['transaction_date'].dt.date >= start_date) & 
        (transactions['transaction_date'].dt.date <= end_date)
    ]
    
    if filtered_transactions.empty:
        st.warning("No transactions found in the selected date range.")
        return
    
    # Enhanced Key Metrics with Real Data Validation
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_transactions['total_amount'].sum()
        st.metric("💰 Total Revenue", format_currency(total_revenue), 
                 help="Calculated from actual transaction data")
    
    with col2:
        total_transactions = len(filtered_transactions)
        st.metric("🧾 Transactions", f"{total_transactions:,}",
                 help="Count of completed sales transactions")
    
    with col3:
        avg_transaction_value = filtered_transactions['total_amount'].mean()
        st.metric("📊 Avg Transaction", format_currency(avg_transaction_value),
                 help="Average value per transaction")
    
    with col4:
        total_items_sold = filtered_transactions['quantity'].sum()
        st.metric("📦 Items Sold", f"{total_items_sold:,}",
                 help="Total quantity of items sold")
    
    # Product Analysis Section
    st.subheader("🏪 Product Performance Analysis")
    
    # Show inventory vs sales comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📦 Available Inventory Products:**")
        if not inventory.empty:
            inventory_summary = inventory.groupby('category').agg({
                'product_name': 'count',
                'stock_quantity': 'sum',
                'unit_price': 'mean'
            }).round(2)
            inventory_summary.columns = ['Product Count', 'Total Stock', 'Avg Price']
            st.dataframe(inventory_summary, use_container_width=True)
        else:
            st.info("No inventory data available")
    
    with col2:
        st.markdown("**🛒 Transaction Products:**")
        transaction_products = filtered_transactions.groupby('category').agg({
            'product_name': 'nunique',
            'quantity': 'sum',
            'total_amount': 'sum'
        }).round(2)
        transaction_products.columns = ['Unique Products', 'Qty Sold', 'Revenue']
        transaction_products['Revenue'] = transaction_products['Revenue'].apply(format_currency)
        st.dataframe(transaction_products, use_container_width=True)
    
    # Enhanced Analytics Charts
    st.subheader("📈 Sales Performance Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Daily Sales Trend")
        daily_sales = filtered_transactions.groupby('transaction_date').agg({
            'total_amount': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        fig = px.line(daily_sales, x='transaction_date', y='total_amount',
                     title=f"Daily Revenue Trend ({format_currency(daily_sales['total_amount'].sum())} Total)",
                     labels={'total_amount': 'Revenue (LSL)', 'transaction_date': 'Date'})
        fig.update_traces(line_color='#1f77b4', line_width=3)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Top Products by Revenue")
        top_products = filtered_transactions.groupby('product_name').agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).sort_values('total_amount', ascending=False).head(10)
        
        fig = px.bar(top_products.reset_index(), 
                    x='total_amount', y='product_name', 
                    orientation='h',
                    title=f"Top {len(top_products)} Products by Revenue",
                    labels={'total_amount': 'Revenue (LSL)', 'product_name': 'Product'})
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional Analytics Row
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 Sales by Category")
        category_sales = filtered_transactions.groupby('category')['total_amount'].sum().reset_index()
        category_sales = category_sales.sort_values('total_amount', ascending=False)
        
        fig = px.pie(category_sales, values='total_amount', names='category',
                    title="Revenue Distribution by Category",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.subheader("📦 Quantity vs Revenue")
        product_performance = filtered_transactions.groupby('product_name').agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).reset_index()
        
        fig = px.scatter(product_performance, 
                        x='quantity', y='total_amount',
                        hover_name='product_name',
                        title="Product Performance: Quantity vs Revenue",
                        labels={'quantity': 'Total Quantity Sold', 'total_amount': 'Total Revenue (LSL)'})
        fig.update_traces(marker=dict(size=12, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Product Analysis Table
    st.subheader("📋 Detailed Product Performance")
    
    # Calculate comprehensive product metrics
    product_analysis = filtered_transactions.groupby('product_name').agg({
        'quantity': 'sum',
        'total_amount': 'sum',
        'unit_price': 'mean',
        'transaction_id': 'count'
    }).round(2)
    
    product_analysis.columns = ['Total Qty Sold', 'Total Revenue', 'Avg Unit Price', 'Times Sold']
    product_analysis['Revenue per Unit'] = (product_analysis['Total Revenue'] / product_analysis['Total Qty Sold']).round(2)
    product_analysis = product_analysis.sort_values('Total Revenue', ascending=False)
    
    # Format currency columns
    currency_cols = ['Total Revenue', 'Avg Unit Price', 'Revenue per Unit']
    for col in currency_cols:
        if col in product_analysis.columns:
            product_analysis[f'{col} (Formatted)'] = product_analysis[col].apply(format_currency)
    
    # Display formatted table
    display_cols = ['Total Qty Sold', 'Total Revenue (Formatted)', 'Avg Unit Price (Formatted)', 
                   'Revenue per Unit (Formatted)', 'Times Sold']
    available_cols = [col for col in display_cols if col in product_analysis.columns]
    
    st.dataframe(
        product_analysis[available_cols].head(15),
        use_container_width=True,
        height=400
    )
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Best Selling Product", 
                 product_analysis.index[0] if len(product_analysis) > 0 else "N/A",
                 help="Product with highest revenue")
    
    with col2:
        most_frequent = product_analysis.sort_values('Times Sold', ascending=False).index[0] if len(product_analysis) > 0 else "N/A"
        st.metric("🔄 Most Frequent Product", most_frequent,
                 help="Product sold in most transactions")
    
    with col3:
        highest_value = product_analysis.sort_values('Avg Unit Price', ascending=False).index[0] if len(product_analysis) > 0 else "N/A"
        st.metric("💎 Highest Value Product", highest_value,
                 help="Product with highest average unit price")
    
    # Inventory vs Sales Analysis
    if not inventory.empty:
        st.subheader("🔄 Inventory vs Sales Analysis")
        
        # Get products that are in both inventory and transactions
        inventory_products = set(inventory['product_name'].str.lower().str.strip())
        transaction_products = set(filtered_transactions['product_name'].str.lower().str.strip())
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📦 Products in Inventory", len(inventory_products))
            
        with col2:
            st.metric("🛒 Products Sold", len(transaction_products))
            
        with col3:
            matching_products = inventory_products.intersection(transaction_products)
            st.metric("🎯 Matching Products", len(matching_products))
        
        # Show product alignment status
        if len(matching_products) > 0:
            st.success(f"✅ Found {len(matching_products)} products that appear in both inventory and sales!")
            st.write("**Matching Products:**", ", ".join(list(matching_products)[:10]))
        else:
            st.warning("⚠️ No direct product matches found between inventory and sales data.")
            st.info("💡 **Recommendation:** Update inventory to include actual sold products or align transaction data with current inventory.")
        
        # Show inventory that hasn't been sold
        unsold_products = inventory_products - transaction_products
        if len(unsold_products) > 0:
            with st.expander(f"📋 View Unsold Inventory Products ({len(unsold_products)} items)"):
                st.write("Products in inventory but not in sales transactions:")
                for product in list(unsold_products)[:20]:  # Show first 20
                    inventory_item = inventory[inventory['product_name'].str.lower().str.strip() == product].iloc[0]
                    st.write(f"• {inventory_item['product_name']} - Stock: {inventory_item['stock_quantity']} @ {format_currency(inventory_item['unit_price'])}")
        
        # Show sales without inventory
        products_without_inventory = transaction_products - inventory_products
        if len(products_without_inventory) > 0:
            with st.expander(f"🛒 View Products Sold Without Inventory ({len(products_without_inventory)} items)"):
                st.write("Products sold but not found in current inventory:")
                for product in list(products_without_inventory)[:20]:  # Show first 20
                    sales_data = filtered_transactions[filtered_transactions['product_name'].str.lower().str.strip() == product]
                    total_sold = sales_data['quantity'].sum()
                    total_revenue = sales_data['total_amount'].sum()
                    st.write(f"• {product.title()} - Sold: {total_sold} units, Revenue: {format_currency(total_revenue)}")

def customer_insights():
    """Customer insights interface"""
    st.subheader("👥 Customer Insights")
    
    # Load data
    inventory, transactions, customers = load_data()
    
    if transactions.empty:
        st.warning("No transaction data available.")
        return
    
    # Customer analysis
    customer_analysis = transactions.groupby('customer_id').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'transaction_date': ['min', 'max']
    }).round(2)
    
    customer_analysis.columns = ['Total Spent', 'Avg Transaction', 'Transaction Count', 'First Purchase', 'Last Purchase']
    customer_analysis = customer_analysis.sort_values('Total Spent', ascending=False)
    
    # Top customers
    st.subheader("🏆 Top Customers")
    top_customers = customer_analysis.head(10)
    st.dataframe(top_customers, width='stretch')
    
    # Customer segments
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Distribution")
        # RFM-like analysis
        customer_analysis['Segment'] = pd.cut(customer_analysis['Total Spent'], 
                                            bins=3, labels=['Bronze', 'Silver', 'Gold'])
        segment_counts = customer_analysis['Segment'].value_counts()
        
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title="Customer Segments by Spending")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Purchase Frequency")
        freq_dist = customer_analysis['Transaction Count'].value_counts().sort_index()
        
        fig = px.bar(x=freq_dist.index, y=freq_dist.values,
                    title="Customer Purchase Frequency",
                    labels={'x': 'Number of Purchases', 'y': 'Number of Customers'})
        st.plotly_chart(fig, width='stretch')

def ai_insights():
    """AI-powered insights interface"""
    st.subheader("🤖 AI-Powered Insights")
    
    # Load AI models
    models = load_ai_models()
    
    # Load data
    inventory, transactions, customers = load_data()
    
    if transactions.empty:
        st.warning("No transaction data available for AI insights.")
        return
    
    # Sales Prediction (Statistical-based)
    st.subheader("📈 Sales Predictions")
    
    col1, col2 = st.columns(2)
    with col1:
        prediction_days = st.slider("Predict for next N days", 1, 30, 7)
    
    try:
        # Statistical-based prediction using historical trends
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        
        # Calculate daily sales averages
        daily_sales = transactions.groupby('transaction_date')['total_amount'].sum()
        recent_avg = daily_sales.tail(7).mean()  # Last 7 days average
        overall_avg = daily_sales.mean()
        trend_factor = recent_avg / overall_avg if overall_avg > 0 else 1
        
        # Generate predictions based on trends
        future_dates = pd.date_range(
            start=transactions['transaction_date'].max() + timedelta(days=1),
            periods=prediction_days,
            freq='D'
        )
        
        predictions = []
        for i, date in enumerate(future_dates):
            # Add some seasonal variation
            day_of_week = date.dayofweek
            weekend_factor = 1.3 if day_of_week in [5, 6] else 1.0
            
            predicted_sales = recent_avg * trend_factor * weekend_factor
            # Add slight random variation
            predicted_sales *= (0.9 + 0.2 * (i % 3) / 3)  # ±10% variation
            
            predictions.append({'date': date, 'predicted_sales': predicted_sales})
        
        pred_df = pd.DataFrame(predictions)
        
        # Plot predictions
        fig = px.line(pred_df, x='date', y='predicted_sales',
                     title=f"Sales Forecast for Next {prediction_days} Days (Trend-based)")
        st.plotly_chart(fig, width='stretch')
        
        # Show prediction summary
        total_predicted = pred_df['predicted_sales'].sum()
        st.info(f"📊 Total predicted revenue for next {prediction_days} days: {format_currency(total_predicted)}")
        
    except Exception as e:
        st.error(f"Error generating sales predictions: {e}")
    
    # Customer Segmentation (RFM-based)
    st.subheader("👥 Customer Segmentation")
    
    try:
        # Calculate RFM metrics
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        current_date = transactions['transaction_date'].max()
        
        rfm_data = transactions.groupby('customer_id').agg({
            'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
            'total_amount': ['count', 'sum']  # Frequency and Monetary
        }).round(2)
        
        rfm_data.columns = ['Recency', 'Frequency', 'Monetary']
        rfm_data = rfm_data.reset_index()
        
        # Simple segmentation based on RFM quartiles
        rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], 4, labels=[4, 3, 2, 1])
        rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
        rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'], 4, labels=[1, 2, 3, 4])
        
        # Create segments
        rfm_data['RFM_Score'] = rfm_data['R_Score'].astype(str) + rfm_data['F_Score'].astype(str) + rfm_data['M_Score'].astype(str)
        
        # Classify into segments
        def classify_customers(row):
            if row['RFM_Score'] in ['444', '443', '434', '344']:
                return 'Champions'
            elif row['RFM_Score'] in ['343', '334', '333', '342']:
                return 'Loyal Customers'
            elif row['RFM_Score'] in ['431', '341', '331', '321']:
                return 'Potential Loyalists'
            elif row['RFM_Score'] in ['241', '231', '221', '142']:
                return 'At Risk'
            else:
                return 'Others'
        
        rfm_data['Segment'] = rfm_data.apply(classify_customers, axis=1)
        
        # Display segment distribution
        segment_counts = rfm_data['Segment'].value_counts()
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title="Customer Segment Distribution (RFM Analysis)")
        st.plotly_chart(fig, width='stretch')
        
        # Show segment details
        segment_summary = rfm_data.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean', 
            'Monetary': 'mean'
        }).round(2)
        
        st.subheader("Segment Summary")
        st.dataframe(segment_summary, width='stretch')
        
        # Show top customers
        st.subheader("Top Customers by Segment")
        top_customers = rfm_data.nlargest(10, 'Monetary')[['customer_id', 'Segment', 'Recency', 'Frequency', 'Monetary']]
        st.dataframe(top_customers, width='stretch')
        
    except Exception as e:
        st.error(f"Error in customer segmentation: {e}")
    
    # Fraud Detection (Rule-based)
    st.subheader("🛡️ Fraud Detection")
    
    try:
        # Rule-based fraud detection
        recent_transactions = transactions.tail(20).copy()
        
        # Calculate fraud risk based on business rules
        def calculate_fraud_risk(row):
            risk_score = 0
            
            # High amount transactions
            if row['total_amount'] > transactions['total_amount'].quantile(0.95):
                risk_score += 0.3
            
            # Large quantity purchases
            if row['quantity'] > transactions['quantity'].quantile(0.9):
                risk_score += 0.2
            
            # Large discount (might indicate employee fraud)
            if row['discount_amount'] > row['subtotal'] * 0.2:
                risk_score += 0.3
            
            # Cash transactions above certain threshold
            if row['payment_method'] == 'Cash' and row['total_amount'] > 1000:
                risk_score += 0.2
            
            return min(risk_score, 1.0)  # Cap at 1.0
        
        recent_transactions['fraud_risk'] = recent_transactions.apply(calculate_fraud_risk, axis=1)
        recent_transactions['risk_level'] = pd.cut(
            recent_transactions['fraud_risk'], 
            bins=[-0.1, 0.3, 0.6, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Display fraud risk distribution
        risk_counts = recent_transactions['risk_level'].value_counts()
        fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                    title="Fraud Risk Distribution (Recent 20 Transactions)",
                    color=risk_counts.index,
                    color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
        st.plotly_chart(fig, width='stretch')
        
        # Show high-risk transactions
        high_risk = recent_transactions[recent_transactions['risk_level'] == 'High']
        if not high_risk.empty:
            st.warning("⚠️ High-risk transactions detected!")
            st.dataframe(high_risk[['transaction_id', 'customer_id', 'total_amount', 'payment_method', 'fraud_risk']], 
                       width='stretch')
        else:
            st.success("✅ No high-risk transactions detected in recent activity")
        
        # Fraud detection summary
        avg_risk = recent_transactions['fraud_risk'].mean()
        st.info(f"📊 Average fraud risk score: {avg_risk:.2f}")
        
        # Show fraud detection rules
        with st.expander("🔍 Fraud Detection Rules"):
            st.write("""
            **Current fraud detection rules:**
            - High amount transactions (>95th percentile): +30% risk
            - Large quantity purchases (>90th percentile): +20% risk  
            - Large discounts (>20% of subtotal): +30% risk
            - Cash transactions >$1000: +20% risk
            """)
            
    except Exception as e:
        st.error(f"Error in fraud detection: {e}")
    
    # Additional AI Insights
    st.subheader("🧠 Business Intelligence Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Product Performance")
        try:
            # Product performance analysis
            product_performance = transactions.groupby('product_name').agg({
                'total_amount': 'sum',
                'quantity': 'sum',
                'transaction_id': 'count'
            }).sort_values('total_amount', ascending=False)
            product_performance.columns = ['Revenue', 'Units Sold', 'Transactions']
            
            st.dataframe(product_performance.head(10), width='stretch')
            
        except Exception as e:
            st.error(f"Error in product analysis: {e}")
    
    with col2:
        st.subheader("💡 Recommendations")
        try:
            # Generate business recommendations
            recommendations = []
            
            # Low stock recommendations
            if not inventory.empty:
                low_stock = inventory[inventory['stock_quantity'] <= inventory['reorder_level']]
                if not low_stock.empty:
                    recommendations.append(f"🔄 Reorder {len(low_stock)} products running low on stock")
            
            # Sales recommendations
            if not transactions.empty:
                avg_transaction = transactions['total_amount'].mean()
                recent_avg = transactions.tail(10)['total_amount'].mean()
                if recent_avg < avg_transaction * 0.9:
                    recommendations.append("📈 Recent sales below average - consider promotions")
                
                # Payment method insights
                payment_dist = transactions['payment_method'].value_counts()
                if 'Cash' in payment_dist and payment_dist['Cash'] > len(transactions) * 0.5:
                    recommendations.append("💳 Consider promoting digital payment methods")
            
            # Customer recommendations
            if not transactions.empty:
                customer_freq = transactions.groupby('customer_id').size()
                repeat_customers = (customer_freq > 1).sum()
                total_customers = customer_freq.count()
                repeat_rate = repeat_customers / total_customers if total_customers > 0 else 0
                
                if repeat_rate < 0.3:
                    recommendations.append("👥 Focus on customer retention programs")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ All key metrics look healthy!")
                
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")

def main():
    """Main application"""
    
    # Sidebar navigation
    st.sidebar.title("🏪 Horizon POS System")
    st.sidebar.markdown("---")
    
    # User role selection
    user_role = st.sidebar.selectbox(
        "Select Your Role",
        ["👔 Manager", "🛒 Sales Assistant", "📊 Dashboard"]
    )
    
    # Navigation based on role
    if user_role == "📊 Dashboard":
        main_dashboard()
    elif user_role == "🛒 Sales Assistant":
        sales_interface()
    elif user_role == "👔 Manager":
        manager_interface()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")
    st.sidebar.info("""
    **Horizon AI POS System**
    
    Features:
    - 🤖 AI-Powered Insights
    - 📦 Inventory Management
    - 🛒 Sales Processing
    - 📊 Real-time Analytics
    - 🛡️ Fraud Detection
    """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 Horizon Enterprise - AI POS System | Lesotho")

if __name__ == "__main__":
    main()