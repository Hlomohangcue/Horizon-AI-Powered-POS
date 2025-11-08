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
    
    # Key Metrics Row
    if not transactions.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = transactions['total_amount'].sum()
            st.metric("💰 Total Sales", f"${total_sales:,.2f}")
        
        with col2:
            total_transactions = len(transactions)
            st.metric("🧾 Total Transactions", f"{total_transactions:,}")
        
        with col3:
            avg_transaction = transactions['total_amount'].mean()
            st.metric("📊 Avg Transaction", f"${avg_transaction:.2f}")
        
        with col4:
            unique_customers = transactions['customer_id'].nunique()
            st.metric("👥 Unique Customers", f"{unique_customers:,}")
    
    # Charts Row
    if not transactions.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Daily Sales Trend")
            daily_sales = transactions.groupby('transaction_date')['total_amount'].sum().reset_index()
            fig = px.line(daily_sales, x='transaction_date', y='total_amount', 
                         title="Daily Sales Revenue")
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("🏷️ Sales by Category")
            category_sales = transactions.groupby('category')['total_amount'].sum().reset_index()
            fig = px.pie(category_sales, values='total_amount', names='category',
                        title="Revenue by Product Category")
            st.plotly_chart(fig, width='stretch')
    
    # Inventory Status
    if not inventory.empty:
        st.subheader("📦 Inventory Status")
        
        # Low stock alerts
        low_stock = inventory[inventory['stock_quantity'] <= inventory['reorder_level']]
        if not low_stock.empty:
            st.warning(f"⚠️ {len(low_stock)} products are low in stock!")
            st.dataframe(low_stock[['product_name', 'stock_quantity', 'reorder_level']])
        
        # Inventory overview
        col1, col2, col3 = st.columns(3)
        with col1:
            total_products = len(inventory)
            st.metric("📦 Total Products", total_products)
        with col2:
            total_stock_value = (inventory['unit_price'] * inventory['stock_quantity']).sum()
            st.metric("💎 Stock Value", f"${total_stock_value:,.2f}")
        with col3:
            out_of_stock = len(inventory[inventory['stock_quantity'] == 0])
            st.metric("🚫 Out of Stock", out_of_stock)

def sales_interface():
    """Sales assistant interface"""
    st.markdown('<h1 class="main-header">🛒 Sales Interface</h1>', unsafe_allow_html=True)
    
    # Load data
    inventory, transactions, customers = load_data()
    
    if inventory.empty:
        st.warning("No inventory data available. Please add products first in the Manager Interface.")
        return
    
    # Sales form
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
                st.info(f"Price: ${product_data['unit_price']:.2f} | Stock: {product_data['stock_quantity']} units")
        
        with col2:
            # Transaction details
            quantity = st.number_input("Quantity", min_value=1, max_value=int(product_data['stock_quantity']) if selected_product else 1)
            payment_method = st.selectbox("Payment Method", ["Cash", "Credit Card", "Debit Card", "Mobile Payment"])
            discount = st.number_input("Discount %", min_value=0.0, max_value=50.0, value=0.0)
            
            # Calculate totals
            if selected_product:
                subtotal = product_data['unit_price'] * quantity
                discount_amount = subtotal * (discount / 100)
                total_amount = subtotal - discount_amount
                
                st.markdown(f"""
                **Order Summary:**
                - Subtotal: ${subtotal:.2f}
                - Discount: ${discount_amount:.2f} ({discount}%)
                - **Total: ${total_amount:.2f}**
                """)
        
        # Submit transaction
        submitted = st.form_submit_button("🧾 Process Sale", type="primary")
        
        if submitted and selected_product:
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
                'subtotal': subtotal,
                'discount_amount': discount_amount,
                'total_amount': total_amount,
                'payment_method': payment_method,
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
                st.markdown(f"""
                **Horizon Enterprise**  
                Transaction ID: {transaction_data['transaction_id']}  
                Date: {transaction_data['transaction_timestamp']}  
                Customer: {customer_name} ({customer_id})  
                
                **Items:**
                - {selected_product} x {quantity} @ ${product_data['unit_price']:.2f} = ${subtotal:.2f}
                - Discount: -${discount_amount:.2f}
                
                **Total: ${total_amount:.2f}**  
                Payment: {payment_method}  
                Cashier: {transaction_data['cashier']}
                
                Thank you for shopping with Horizon Enterprise!
                """)
                
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
    
    # Add new product
    with st.expander("➕ Add New Product"):
        with st.form("add_product_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                product_name = st.text_input("Product Name")
                category = st.selectbox("Category", ["Electronics", "Clothing", "Food", "Books", "Home", "Sports", "Other"])
                unit_price = st.number_input("Unit Price ($)", min_value=0.01, value=10.00)
            
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
    
    # Current inventory
    if not inventory.empty:
        st.subheader("Current Inventory")
        
        # Search and filter
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("🔍 Search Products")
        with col2:
            category_filter = st.selectbox("Filter by Category", ["All"] + list(inventory['category'].unique()))
        with col3:
            stock_filter = st.selectbox("Stock Status", ["All", "In Stock", "Low Stock", "Out of Stock"])
        
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
        
        # Display inventory with edit capability
        st.dataframe(
            filtered_inventory,
            width='stretch',
            hide_index=True
        )
        
        # Bulk actions
        with st.expander("⚙️ Bulk Actions"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Update Stock")
                selected_product = st.selectbox("Select Product", filtered_inventory['product_name'].tolist())
                new_quantity = st.number_input("New Quantity", min_value=0)
                
                if st.button("Update Stock"):
                    inventory.loc[inventory['product_name'] == selected_product, 'stock_quantity'] = new_quantity
                    if save_inventory(inventory):
                        st.success(f"✅ Stock updated for {selected_product}")
                        st.cache_data.clear()
            
            with col2:
                st.subheader("Price Update")
                price_product = st.selectbox("Select Product for Price Update", filtered_inventory['product_name'].tolist())
                new_price = st.number_input("New Price ($)", min_value=0.01)
                
                if st.button("Update Price"):
                    inventory.loc[inventory['product_name'] == price_product, 'unit_price'] = new_price
                    if save_inventory(inventory):
                        st.success(f"✅ Price updated for {price_product}")
                        st.cache_data.clear()

def sales_analytics():
    """Sales analytics interface"""
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
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_transactions['total_amount'].sum()
        st.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        total_transactions = len(filtered_transactions)
        st.metric("🧾 Transactions", f"{total_transactions:,}")
    
    with col3:
        avg_transaction_value = filtered_transactions['total_amount'].mean()
        st.metric("📊 Avg Transaction", f"${avg_transaction_value:.2f}")
    
    with col4:
        total_items_sold = filtered_transactions['quantity'].sum()
        st.metric("📦 Items Sold", f"{total_items_sold:,}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Sales Trend")
        daily_sales = filtered_transactions.groupby('transaction_date')['total_amount'].sum().reset_index()
        fig = px.line(daily_sales, x='transaction_date', y='total_amount',
                     title="Daily Revenue")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Top Products")
        top_products = filtered_transactions.groupby('product_name').agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).sort_values('total_amount', ascending=False).head(10)
        
        fig = px.bar(top_products.reset_index(), x='product_name', y='total_amount',
                    title="Top 10 Products by Revenue")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, width='stretch')
    
    # Category analysis
    st.subheader("Category Performance")
    category_performance = filtered_transactions.groupby('category').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'quantity': 'sum'
    }).round(2)
    category_performance.columns = ['Total Revenue', 'Avg Transaction', 'Transaction Count', 'Total Quantity']
    st.dataframe(category_performance, width='stretch')

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
    
    if not models:
        st.warning("AI models are not available. Please train the models first.")
        return
    
    # Load data
    inventory, transactions, customers = load_data()
    
    if transactions.empty:
        st.warning("No transaction data available for AI insights.")
        return
    
    # Sales Prediction
    if 'sales_predictor' in models:
        st.subheader("📈 Sales Predictions")
        
        col1, col2 = st.columns(2)
        with col1:
            prediction_days = st.slider("Predict for next N days", 1, 30, 7)
        
        try:
            # Generate prediction data
            future_dates = pd.date_range(
                start=transactions['transaction_date'].max() + timedelta(days=1),
                periods=prediction_days,
                freq='D'
            )
            
            predictions = []
            for date in future_dates:
                # Create sample data for prediction
                sample_data = pd.DataFrame({
                    'transaction_date': [date],
                    'product_category': ['Electronics'],  # Default category
                    'unit_price': [transactions['unit_price'].mean()],
                    'customer_id': [1]
                })
                
                pred = models['sales_predictor'].predict_sales(sample_data)
                predictions.append({'date': date, 'predicted_sales': pred[0] if len(pred) > 0 else 0})
            
            pred_df = pd.DataFrame(predictions)
            
            # Plot predictions
            fig = px.line(pred_df, x='date', y='predicted_sales',
                         title=f"Sales Forecast for Next {prediction_days} Days")
            st.plotly_chart(fig, width='stretch')
            
        except Exception as e:
            st.error(f"Error generating sales predictions: {e}")
    
    # Customer Segmentation
    if 'customer_segmentation' in models:
        st.subheader("👥 Customer Segmentation")
        
        try:
            # Get customer segments
            customer_data = transactions.groupby('customer_id').agg({
                'total_amount': 'sum',
                'transaction_date': 'count'
            }).reset_index()
            
            segments = models['customer_segmentation'].predict_segment(customer_data)
            customer_data['segment'] = segments
            
            # Display segment distribution
            segment_counts = pd.Series(segments).value_counts()
            fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                        title="Customer Segment Distribution")
            st.plotly_chart(fig, width='stretch')
            
            # Show segment details
            st.dataframe(customer_data.head(20), width='stretch')
            
        except Exception as e:
            st.error(f"Error in customer segmentation: {e}")
    
    # Fraud Detection
    if 'fraud_detector' in models:
        st.subheader("🛡️ Fraud Detection")
        
        try:
            # Check recent transactions for fraud
            recent_transactions = transactions.tail(20)
            fraud_scores = models['fraud_detector'].detect_fraud(recent_transactions)
            
            recent_transactions['fraud_risk'] = fraud_scores
            recent_transactions['risk_level'] = pd.cut(
                recent_transactions['fraud_risk'], 
                bins=[0, 0.3, 0.7, 1.0], 
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
                st.dataframe(high_risk[['transaction_id', 'customer_id', 'total_amount', 'fraud_risk']], 
                           width='stretch')
            else:
                st.success("✅ No high-risk transactions detected")
                
        except Exception as e:
            st.error(f"Error in fraud detection: {e}")

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
    st.sidebar.caption("© 2025 Horizon Enterprise - AI POS System")

if __name__ == "__main__":
    main()