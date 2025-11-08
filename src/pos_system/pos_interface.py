"""
POS Interface - Main User Interface for Horizon AI-Powered POS System
=====================================================================

This module provides the main user interface for the Point of Sale system,
integrating all AI models and providing real-time insights and recommendations.

Author: [Your Name]
Course: AI for Software Engineering
Date: November 8, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class POSInterface:
    """
    Main POS Interface that integrates all AI models
    
    This class provides the user interface for the POS system and coordinates
    between different AI models to provide intelligent insights.
    """
    
    def __init__(self, sales_predictor=None, customer_segmentation=None, fraud_detector=None):
        """
        Initialize POS Interface with AI models
        
        Args:
            sales_predictor: Trained sales prediction model
            customer_segmentation: Trained customer segmentation model  
            fraud_detector: Trained fraud detection model
        """
        self.sales_predictor = sales_predictor
        self.customer_segmentation = customer_segmentation
        self.fraud_detector = fraud_detector
        
        # Current session data
        self.current_transaction = {}
        self.daily_transactions = []
        self.customer_database = {}
        self.inventory = self._initialize_sample_inventory()
        
        # System statistics
        self.session_stats = {
            'transactions_processed': 0,
            'total_revenue': 0.0,
            'fraud_alerts': 0,
            'session_start': datetime.now()
        }
        
        logger.info("POS Interface initialized with AI models")
    
    def _initialize_sample_inventory(self):
        """Initialize sample inventory for demonstration"""
        return {
            'ELEC001': {'name': 'Smartphone', 'category': 'Electronics', 'price': 599.99, 'stock': 50},
            'ELEC002': {'name': 'Laptop', 'category': 'Electronics', 'price': 999.99, 'stock': 25},
            'CLOTH001': {'name': 'T-Shirt', 'category': 'Clothing', 'price': 29.99, 'stock': 100},
            'CLOTH002': {'name': 'Jeans', 'category': 'Clothing', 'price': 79.99, 'stock': 60},
            'FOOD001': {'name': 'Coffee', 'category': 'Food', 'price': 4.99, 'stock': 200},
            'FOOD002': {'name': 'Sandwich', 'category': 'Food', 'price': 8.99, 'stock': 80},
            'JEWEL001': {'name': 'Gold Ring', 'category': 'Jewelry', 'price': 1299.99, 'stock': 5},
            'BOOK001': {'name': 'AI Textbook', 'category': 'Books', 'price': 89.99, 'stock': 30}
        }
    
    def start(self):
        """
        Start the main POS interface
        
        This method runs the main interaction loop for the POS system
        """
        print("🏪 Welcome to Horizon AI-Powered POS System!")
        print("=" * 60)
        
        while True:
            try:
                self._display_main_menu()
                choice = input("\nEnter your choice (1-8): ").strip()
                
                if choice == '1':
                    self._process_transaction()
                elif choice == '2':
                    self._view_sales_predictions()
                elif choice == '3':
                    self._analyze_customer_segments()
                elif choice == '4':
                    self._check_fraud_alerts()
                elif choice == '5':
                    self._view_inventory_status()
                elif choice == '6':
                    self._view_session_statistics()
                elif choice == '7':
                    self._generate_ai_insights()
                elif choice == '8':
                    print("\n👋 Thank you for using Horizon AI-Powered POS System!")
                    print("Session ended at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    break
                else:
                    print("❌ Invalid choice. Please enter a number between 1-8.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 System shutdown initiated by user.")
                break
            except Exception as e:
                logger.error(f"Error in main interface: {str(e)}")
                print(f"❌ Error: {str(e)}")
    
    def _display_main_menu(self):
        """Display the main menu options"""
        print("\n" + "=" * 60)
        print("🤖 HORIZON AI-POWERED POS SYSTEM - MAIN MENU")
        print("=" * 60)
        print("1. 🛒 Process New Transaction")
        print("2. 📈 View Sales Predictions") 
        print("3. 👥 Analyze Customer Segments")
        print("4. 🛡️  Check Fraud Alerts")
        print("5. 📦 View Inventory Status")
        print("6. 📊 View Session Statistics")
        print("7. 🧠 Generate AI Insights")
        print("8. 🚪 Exit System")
        print("=" * 60)
    
    def _process_transaction(self):
        """Process a new transaction with AI analysis"""
        print("\n🛒 PROCESSING NEW TRANSACTION")
        print("-" * 40)
        
        try:
            # Get transaction details
            transaction = self._get_transaction_input()
            
            if not transaction:
                return
            
            # Real-time fraud detection
            fraud_result = self._check_transaction_fraud(transaction)
            
            if fraud_result['risk_level'] in ['HIGH', 'MEDIUM']:
                print(f"\n⚠️  FRAUD ALERT: {fraud_result['risk_level']} RISK DETECTED!")
                print(f"Risk Score: {fraud_result['overall_risk_score']:.2f}")
                print("Reasons:", ", ".join(fraud_result['explanations']))
                print(f"Recommended Action: {fraud_result['recommended_action']}")
                
                if fraud_result['risk_level'] == 'HIGH':
                    confirm = input("\nTransaction flagged as HIGH RISK. Proceed anyway? (y/N): ")
                    if confirm.lower() != 'y':
                        print("❌ Transaction cancelled due to fraud risk.")
                        self.session_stats['fraud_alerts'] += 1
                        return
            
            # Process the transaction
            self._complete_transaction(transaction)
            
            # Get AI recommendations
            self._provide_ai_recommendations(transaction)
            
        except Exception as e:
            logger.error(f"Error processing transaction: {str(e)}")
            print(f"❌ Error processing transaction: {str(e)}")
    
    def _get_transaction_input(self):
        """Get transaction details from user input"""
        print("Enter transaction details:")
        
        try:
            # Customer information
            customer_id = input("Customer ID (or 'new' for new customer): ").strip()
            if customer_id.lower() == 'new':
                customer_id = f"CUST_{len(self.customer_database) + 1:05d}"
                print(f"New customer ID assigned: {customer_id}")
            
            # Product selection
            print("\nAvailable Products:")
            for prod_id, details in self.inventory.items():
                if details['stock'] > 0:
                    print(f"  {prod_id}: {details['name']} - ${details['price']} ({details['stock']} in stock)")
            
            product_id = input("\nEnter Product ID: ").strip().upper()
            
            if product_id not in self.inventory:
                print("❌ Invalid product ID.")
                return None
            
            if self.inventory[product_id]['stock'] <= 0:
                print("❌ Product out of stock.")
                return None
            
            quantity = int(input("Enter quantity: "))
            
            if quantity <= 0:
                print("❌ Invalid quantity.")
                return None
            
            if quantity > self.inventory[product_id]['stock']:
                print(f"❌ Insufficient stock. Available: {self.inventory[product_id]['stock']}")
                return None
            
            # Payment method
            print("\nPayment Methods: 1) Cash, 2) Credit Card, 3) Debit Card, 4) Online Payment")
            payment_choice = input("Select payment method (1-4): ").strip()
            payment_methods = {'1': 'cash', '2': 'credit_card', '3': 'debit_card', '4': 'online_payment'}
            payment_method = payment_methods.get(payment_choice, 'cash')
            
            # Calculate amounts
            unit_price = self.inventory[product_id]['price']
            total_amount = unit_price * quantity
            
            # Create transaction record
            transaction = {
                'transaction_id': f"TXN_{len(self.daily_transactions) + 1:06d}",
                'customer_id': customer_id,
                'product_id': product_id,
                'product_name': self.inventory[product_id]['name'],
                'product_category': self.inventory[product_id]['category'],
                'quantity': quantity,
                'unit_price': unit_price,
                'total_amount': total_amount,
                'payment_method': payment_method,
                'transaction_timestamp': datetime.now(),
                'transaction_date': datetime.now().date(),
                'store_location': 'Store_A'
            }
            
            print(f"\nTransaction Summary:")
            print(f"  Transaction ID: {transaction['transaction_id']}")
            print(f"  Customer: {customer_id}")
            print(f"  Product: {transaction['product_name']}")
            print(f"  Quantity: {quantity}")
            print(f"  Unit Price: ${unit_price:.2f}")
            print(f"  Total Amount: ${total_amount:.2f}")
            print(f"  Payment: {payment_method.replace('_', ' ').title()}")
            
            return transaction
            
        except ValueError:
            print("❌ Invalid input. Please enter valid numbers.")
            return None
        except Exception as e:
            print(f"❌ Error getting transaction input: {str(e)}")
            return None
    
    def _check_transaction_fraud(self, transaction):
        """Check transaction for fraud using AI model"""
        if not self.fraud_detector or not self.fraud_detector.is_trained:
            # Return default safe result if no fraud detector
            return {
                'overall_risk_score': 0.1,
                'risk_level': 'LOW',
                'explanations': ['Fraud detection not available'],
                'recommended_action': 'ALLOW transaction with standard processing'
            }
        
        try:
            # Convert transaction to DataFrame
            df = pd.DataFrame([transaction])
            
            # Detect fraud
            fraud_results = self.fraud_detector.detect_fraud(df)
            
            # Return first transaction result
            if fraud_results['transactions']:
                return fraud_results['transactions'][0]
            else:
                return {
                    'overall_risk_score': 0.1,
                    'risk_level': 'LOW',
                    'explanations': ['Normal transaction'],
                    'recommended_action': 'ALLOW transaction'
                }
                
        except Exception as e:
            logger.error(f"Error in fraud detection: {str(e)}")
            return {
                'overall_risk_score': 0.2,
                'risk_level': 'LOW',
                'explanations': ['Fraud check failed'],
                'recommended_action': 'ALLOW with caution'
            }
    
    def _complete_transaction(self, transaction):
        """Complete the transaction and update records"""
        try:
            # Update inventory
            product_id = transaction['product_id']
            self.inventory[product_id]['stock'] -= transaction['quantity']
            
            # Add to daily transactions
            self.daily_transactions.append(transaction)
            
            # Update customer database
            customer_id = transaction['customer_id']
            if customer_id not in self.customer_database:
                self.customer_database[customer_id] = {
                    'first_transaction': transaction['transaction_timestamp'],
                    'total_transactions': 0,
                    'total_spent': 0.0,
                    'favorite_categories': {}
                }
            
            customer = self.customer_database[customer_id]
            customer['total_transactions'] += 1
            customer['total_spent'] += transaction['total_amount']
            customer['last_transaction'] = transaction['transaction_timestamp']
            
            # Update favorite categories
            category = transaction['product_category']
            customer['favorite_categories'][category] = customer['favorite_categories'].get(category, 0) + 1
            
            # Update session statistics
            self.session_stats['transactions_processed'] += 1
            self.session_stats['total_revenue'] += transaction['total_amount']
            
            print(f"\n✅ Transaction {transaction['transaction_id']} completed successfully!")
            print(f"💰 Total: ${transaction['total_amount']:.2f}")
            print(f"📦 Remaining stock for {transaction['product_name']}: {self.inventory[product_id]['stock']}")
            
        except Exception as e:
            logger.error(f"Error completing transaction: {str(e)}")
            print(f"❌ Error completing transaction: {str(e)}")
    
    def _provide_ai_recommendations(self, transaction):
        """Provide AI-powered recommendations based on the transaction"""
        print("\n🧠 AI RECOMMENDATIONS:")
        print("-" * 30)
        
        try:
            customer_id = transaction['customer_id']
            
            # Customer segmentation insights
            if self.customer_segmentation and self.customer_segmentation.is_trained:
                customer_data = pd.DataFrame(self.daily_transactions)
                if not customer_data.empty:
                    segments = self.customer_segmentation.predict_segment(customer_data)
                    customer_segment = next((s for s in segments if s['customer_id'] == customer_id), None)
                    
                    if customer_segment:
                        print(f"🎯 Customer Segment: {customer_segment['segment_name']}")
                        print(f"💎 Predicted CLV: ${customer_segment['predicted_clv']:.2f}")
                        print(f"🔥 Loyalty Level: {customer_segment['loyalty_level']}")
                        
                        # Get segment recommendations
                        recommendations = self.customer_segmentation.get_segment_recommendations(customer_segment['segment_name'])
                        print(f"📋 Strategy: {recommendations['strategy']}")
            
            # Product recommendations based on category
            category = transaction['product_category']
            related_products = [p for p, details in self.inventory.items() 
                              if details['category'] == category and p != transaction['product_id'] and details['stock'] > 0]
            
            if related_products:
                print(f"\n🛍️  Customers who bought {transaction['product_name']} also like:")
                for prod_id in related_products[:3]:  # Show top 3
                    product = self.inventory[prod_id]
                    print(f"   • {product['name']} - ${product['price']:.2f}")
            
            # Inventory alerts
            if self.inventory[transaction['product_id']]['stock'] < 10:
                print(f"\n⚠️  LOW STOCK ALERT: {transaction['product_name']} ({self.inventory[transaction['product_id']]['stock']} remaining)")
            
            # Cross-selling opportunities
            customer = self.customer_database[customer_id]
            if customer['total_transactions'] >= 3:
                print(f"\n⭐ LOYALTY REWARD: Customer has made {customer['total_transactions']} purchases!")
                print("   Consider offering a loyalty discount or reward points.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            print("❌ Unable to generate AI recommendations at this time.")
    
    def _view_sales_predictions(self):
        """Display sales predictions from AI model"""
        print("\n📈 SALES PREDICTIONS")
        print("-" * 40)
        
        try:
            if not self.sales_predictor or not self.sales_predictor.is_trained:
                print("❌ Sales prediction model not available or not trained.")
                return
            
            # Generate predictions
            print("Generating sales predictions...")
            
            # Use daily transactions as context
            if self.daily_transactions:
                transaction_df = pd.DataFrame(self.daily_transactions)
                predictions = self.sales_predictor.predict_sales(data=transaction_df, days_ahead=7)
            else:
                predictions = self.sales_predictor.predict_sales(days_ahead=7)
            
            print(f"\n📊 7-Day Sales Forecast:")
            print(f"   Total Predicted Revenue: ${predictions['total_predicted']:.2f}")
            print(f"   Average Daily Revenue: ${predictions['average_daily']:.2f}")
            
            print(f"\n📅 Daily Breakdown:")
            for i, daily_pred in enumerate(predictions['predictions']):
                date = datetime.now() + timedelta(days=i+1)
                print(f"   Day {i+1} ({date.strftime('%Y-%m-%d')}): ${daily_pred:.2f}")
            
            # Feature importance if available
            if hasattr(self.sales_predictor, 'get_feature_importance'):
                importance = self.sales_predictor.get_feature_importance()
                print(f"\n🔍 Key Sales Drivers:")
                for _, row in importance.head(5).iterrows():
                    print(f"   • {row['feature']}: {row['importance']:.3f}")
            
        except Exception as e:
            logger.error(f"Error getting sales predictions: {str(e)}")
            print(f"❌ Error generating sales predictions: {str(e)}")
    
    def _analyze_customer_segments(self):
        """Analyze and display customer segments"""
        print("\n👥 CUSTOMER SEGMENT ANALYSIS")
        print("-" * 40)
        
        try:
            if not self.customer_segmentation or not self.customer_segmentation.is_trained:
                print("❌ Customer segmentation model not available or not trained.")
                return
            
            if not self.daily_transactions:
                print("❌ No transaction data available for segmentation.")
                return
            
            # Get cluster summary
            segment_summary = self.customer_segmentation.get_cluster_summary()
            
            print("🎯 Customer Segments Overview:")
            for idx, row in segment_summary.iterrows():
                print(f"\n📊 Cluster {idx}: {row['segment_name']}")
                print(f"   👥 Size: {row['cluster_size']} customers")
                print(f"   💰 Avg CLV: ${row['clv_mean']:.2f}")
                print(f"   🔄 Avg Frequency: {row['frequency_mean']:.1f} transactions")
                print(f"   📅 Avg Recency: {row['recency_mean']:.0f} days")
            
            # Today's customer analysis
            transaction_df = pd.DataFrame(self.daily_transactions)
            segments = self.customer_segmentation.predict_segment(transaction_df)
            
            print(f"\n📈 Today's Customer Breakdown:")
            segment_counts = {}
            for segment in segments:
                seg_name = segment['segment_name']
                segment_counts[seg_name] = segment_counts.get(seg_name, 0) + 1
            
            for seg_name, count in segment_counts.items():
                print(f"   • {seg_name}: {count} customers")
            
        except Exception as e:
            logger.error(f"Error analyzing customer segments: {str(e)}")
            print(f"❌ Error analyzing customer segments: {str(e)}")
    
    def _check_fraud_alerts(self):
        """Display fraud detection alerts and statistics"""
        print("\n🛡️  FRAUD DETECTION ALERTS")
        print("-" * 40)
        
        try:
            if not self.fraud_detector or not self.fraud_detector.is_trained:
                print("❌ Fraud detection model not available or not trained.")
                return
            
            # Get fraud statistics
            fraud_stats = self.fraud_detector.get_fraud_statistics()
            
            print("📊 Fraud Detection Statistics:")
            print(f"   Model Accuracy: {fraud_stats['model_accuracy']*100:.1f}%")
            print(f"   Detection Rate: {fraud_stats['detection_rate']*100:.1f}%")
            print(f"   False Positive Rate: {fraud_stats['false_positive_rate']*100:.1f}%")
            
            print(f"\n⚠️  Session Fraud Alerts: {self.session_stats['fraud_alerts']}")
            
            print(f"\n🚨 Common Fraud Patterns:")
            for pattern in fraud_stats['most_common_fraud_patterns']:
                print(f"   • {pattern}")
            
            print(f"\n⏰ High-Risk Time Periods:")
            for period in fraud_stats['high_risk_time_periods']:
                print(f"   • {period}")
            
            print(f"\n💳 High-Risk Payment Methods:")
            for method in fraud_stats['high_risk_payment_methods']:
                print(f"   • {method.replace('_', ' ').title()}")
            
            # Analyze today's transactions for fraud
            if self.daily_transactions:
                transaction_df = pd.DataFrame(self.daily_transactions)
                fraud_results = self.fraud_detector.detect_fraud(transaction_df)
                
                high_risk = sum(1 for t in fraud_results['transactions'] if t['risk_level'] == 'HIGH')
                medium_risk = sum(1 for t in fraud_results['transactions'] if t['risk_level'] == 'MEDIUM')
                
                print(f"\n📈 Today's Risk Analysis:")
                print(f"   Total Transactions: {fraud_results['total_transactions']}")
                print(f"   High Risk: {high_risk}")
                print(f"   Medium Risk: {medium_risk}")
                print(f"   Overall Fraud Rate: {fraud_results['fraud_rate']*100:.2f}%")
            
        except Exception as e:
            logger.error(f"Error checking fraud alerts: {str(e)}")
            print(f"❌ Error checking fraud alerts: {str(e)}")
    
    def _view_inventory_status(self):
        """Display current inventory status with AI insights"""
        print("\n📦 INVENTORY STATUS")
        print("-" * 40)
        
        try:
            print("Current Inventory Levels:")
            
            low_stock_items = []
            out_of_stock_items = []
            
            for product_id, details in self.inventory.items():
                stock_status = "✅" if details['stock'] > 20 else "⚠️" if details['stock'] > 5 else "❌"
                print(f"{stock_status} {product_id}: {details['name']} - {details['stock']} units")
                
                if details['stock'] == 0:
                    out_of_stock_items.append(details['name'])
                elif details['stock'] < 10:
                    low_stock_items.append(details['name'])
            
            if low_stock_items:
                print(f"\n⚠️  LOW STOCK ALERTS:")
                for item in low_stock_items:
                    print(f"   • {item}")
            
            if out_of_stock_items:
                print(f"\n❌ OUT OF STOCK:")
                for item in out_of_stock_items:
                    print(f"   • {item}")
            
            # Calculate inventory value
            total_value = sum(details['price'] * details['stock'] for details in self.inventory.values())
            print(f"\n💰 Total Inventory Value: ${total_value:.2f}")
            
            # Best selling categories today
            if self.daily_transactions:
                transaction_df = pd.DataFrame(self.daily_transactions)
                category_sales = transaction_df.groupby('product_category')['quantity'].sum().sort_values(ascending=False)
                
                print(f"\n🏆 Top Selling Categories Today:")
                for category, qty in category_sales.head(3).items():
                    print(f"   • {category}: {qty} units sold")
            
        except Exception as e:
            logger.error(f"Error viewing inventory: {str(e)}")
            print(f"❌ Error viewing inventory: {str(e)}")
    
    def _view_session_statistics(self):
        """Display session statistics and performance metrics"""
        print("\n📊 SESSION STATISTICS")
        print("-" * 40)
        
        try:
            session_duration = datetime.now() - self.session_stats['session_start']
            
            print(f"Session Information:")
            print(f"   Start Time: {self.session_stats['session_start'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Duration: {str(session_duration).split('.')[0]}")
            print(f"   Transactions Processed: {self.session_stats['transactions_processed']}")
            print(f"   Total Revenue: ${self.session_stats['total_revenue']:.2f}")
            print(f"   Fraud Alerts: {self.session_stats['fraud_alerts']}")
            
            if self.session_stats['transactions_processed'] > 0:
                avg_transaction = self.session_stats['total_revenue'] / self.session_stats['transactions_processed']
                print(f"   Average Transaction: ${avg_transaction:.2f}")
            
            print(f"\nCustomer Information:")
            print(f"   Total Customers: {len(self.customer_database)}")
            print(f"   New Customers Today: {sum(1 for c in self.customer_database.values() if c['total_transactions'] == 1)}")
            
            if self.customer_database:
                returning_customers = sum(1 for c in self.customer_database.values() if c['total_transactions'] > 1)
                print(f"   Returning Customers: {returning_customers}")
                
                avg_customer_value = sum(c['total_spent'] for c in self.customer_database.values()) / len(self.customer_database)
                print(f"   Average Customer Value: ${avg_customer_value:.2f}")
            
            # Performance metrics
            if session_duration.total_seconds() > 0:
                transactions_per_hour = (self.session_stats['transactions_processed'] * 3600) / session_duration.total_seconds()
                revenue_per_hour = (self.session_stats['total_revenue'] * 3600) / session_duration.total_seconds()
                
                print(f"\nPerformance Metrics:")
                print(f"   Transactions/Hour: {transactions_per_hour:.1f}")
                print(f"   Revenue/Hour: ${revenue_per_hour:.2f}")
            
        except Exception as e:
            logger.error(f"Error viewing statistics: {str(e)}")
            print(f"❌ Error viewing statistics: {str(e)}")
    
    def _generate_ai_insights(self):
        """Generate comprehensive AI insights and recommendations"""
        print("\n🧠 AI INSIGHTS & RECOMMENDATIONS")
        print("-" * 50)
        
        try:
            if not self.daily_transactions:
                print("❌ No transaction data available for AI analysis.")
                return
            
            transaction_df = pd.DataFrame(self.daily_transactions)
            
            print("📊 BUSINESS INTELLIGENCE SUMMARY:")
            print("=" * 40)
            
            # Sales insights
            total_sales = transaction_df['total_amount'].sum()
            avg_transaction = transaction_df['total_amount'].mean()
            
            print(f"💰 Sales Performance:")
            print(f"   Today's Revenue: ${total_sales:.2f}")
            print(f"   Average Transaction: ${avg_transaction:.2f}")
            print(f"   Transaction Count: {len(transaction_df)}")
            
            # Best performing products
            product_performance = transaction_df.groupby('product_name').agg({
                'quantity': 'sum',
                'total_amount': 'sum'
            }).sort_values('total_amount', ascending=False)
            
            print(f"\n🏆 Top Performing Products:")
            for product, data in product_performance.head(3).iterrows():
                print(f"   • {product}: ${data['total_amount']:.2f} ({data['quantity']} units)")
            
            # Time-based patterns
            transaction_df['hour'] = pd.to_datetime(transaction_df['transaction_timestamp']).dt.hour
            hourly_sales = transaction_df.groupby('hour')['total_amount'].sum()
            peak_hour = hourly_sales.idxmax()
            
            print(f"\n⏰ Time-Based Insights:")
            print(f"   Peak Sales Hour: {peak_hour}:00")
            print(f"   Peak Hour Revenue: ${hourly_sales.max():.2f}")
            
            # Customer insights
            customer_analysis = transaction_df.groupby('customer_id').agg({
                'total_amount': ['sum', 'count', 'mean']
            })
            customer_analysis.columns = ['total_spent', 'transaction_count', 'avg_transaction']
            
            top_customer = customer_analysis.sort_values('total_spent', ascending=False).index[0]
            top_customer_value = customer_analysis.loc[top_customer, 'total_spent']
            
            print(f"\n👑 Customer Insights:")
            print(f"   Top Customer: {top_customer}")
            print(f"   Top Customer Value: ${top_customer_value:.2f}")
            print(f"   Unique Customers: {len(customer_analysis)}")
            
            # Category performance
            category_performance = transaction_df.groupby('product_category')['total_amount'].sum().sort_values(ascending=False)
            
            print(f"\n📂 Category Performance:")
            for category, revenue in category_performance.items():
                percentage = (revenue / total_sales) * 100
                print(f"   • {category}: ${revenue:.2f} ({percentage:.1f}%)")
            
            # AI Recommendations
            print(f"\n🎯 AI RECOMMENDATIONS:")
            print("=" * 30)
            
            # Inventory recommendations
            print("📦 Inventory Management:")
            if peak_hour:
                print(f"   • Schedule staff for peak hour ({peak_hour}:00)")
            
            best_category = category_performance.index[0]
            print(f"   • Focus marketing on {best_category} (top performing category)")
            
            # Check for low stock in high-performing products
            for product, data in product_performance.head(3).iterrows():
                # Find product ID for this product name
                product_id = None
                for pid, details in self.inventory.items():
                    if details['name'] == product:
                        product_id = pid
                        break
                
                if product_id and self.inventory[product_id]['stock'] < 20:
                    print(f"   • URGENT: Restock {product} (high sales, low inventory)")
            
            # Customer retention recommendations
            single_transaction_customers = len([c for c in self.customer_database.values() if c['total_transactions'] == 1])
            if single_transaction_customers > 0:
                print(f"\n👥 Customer Retention:")
                print(f"   • {single_transaction_customers} customers made only 1 purchase")
                print(f"   • Consider follow-up marketing for customer retention")
            
            # Fraud prevention recommendations
            high_value_transactions = len(transaction_df[transaction_df['total_amount'] > 500])
            if high_value_transactions > 0:
                print(f"\n🛡️  Security Recommendations:")
                print(f"   • {high_value_transactions} high-value transactions today")
                print(f"   • Consider additional verification for transactions >$500")
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            print(f"❌ Error generating AI insights: {str(e)}")