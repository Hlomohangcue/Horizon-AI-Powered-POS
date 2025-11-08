"""
Unit Tests for Horizon AI-Powered POS System
============================================

This module contains comprehensive unit tests for all components
of the AI-powered POS system.

Author: [Your Name]
Course: AI for Software Engineering
Date: November 8, 2025
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_models.sales_predictor import SalesPredictor
from ai_models.customer_segmentation import CustomerSegmentation
from ai_models.fraud_detector import FraudDetector
from pos_system.pos_interface import POSInterface

class TestSalesPredictor(unittest.TestCase):
    """Test cases for Sales Predictor model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = SalesPredictor()
        
        # Create sample training data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        self.sample_data = pd.DataFrame({
            'transaction_date': dates,
            'total_amount': np.random.normal(1000, 200, len(dates)),
            'quantity': np.random.normal(50, 10, len(dates)),
            'unit_price': np.random.normal(20, 5, len(dates)),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food'], len(dates)),
            'customer_id': np.random.randint(1, 100, len(dates))
        })
    
    def test_initialization(self):
        """Test sales predictor initialization"""
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.scaler)
        self.assertFalse(self.predictor.is_trained)
    
    def test_feature_preparation(self):
        """Test feature preparation"""
        features = self.predictor.prepare_features(self.sample_data)
        
        # Check that features are created
        self.assertIsNotNone(features)
        self.assertGreater(len(features.columns), len(self.sample_data.columns))
        
        # Check for temporal features
        expected_features = ['year', 'month', 'day_of_week']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
    
    def test_model_training(self):
        """Test model training process"""
        # Train the model
        self.predictor.train(data=self.sample_data)
        
        # Verify model is trained
        self.assertTrue(self.predictor.is_trained)
        self.assertIsNotNone(self.predictor.feature_names)
    
    def test_sales_prediction(self):
        """Test sales prediction functionality"""
        # Train model first
        self.predictor.train(data=self.sample_data)
        
        # Make predictions
        predictions = self.predictor.predict_sales(days_ahead=7)
        
        # Verify prediction structure
        self.assertIn('predictions', predictions)
        self.assertIn('total_predicted', predictions)
        self.assertEqual(len(predictions['predictions']), 7)
        self.assertGreater(predictions['total_predicted'], 0)

class TestCustomerSegmentation(unittest.TestCase):
    """Test cases for Customer Segmentation model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.segmentation = CustomerSegmentation(n_clusters=3)
        
        # Create sample customer transaction data
        self.sample_data = pd.DataFrame({
            'customer_id': np.repeat(range(1, 101), 10),  # 100 customers, 10 transactions each
            'transaction_date': pd.date_range('2024-01-01', periods=1000, freq='D')[:1000],
            'total_amount': np.random.lognormal(3, 1, 1000),
            'quantity': np.random.randint(1, 5, 1000),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food'], 1000)
        })
    
    def test_initialization(self):
        """Test customer segmentation initialization"""
        self.assertEqual(self.segmentation.n_clusters, 3)
        self.assertFalse(self.segmentation.is_trained)
    
    def test_rfm_calculation(self):
        """Test RFM feature calculation"""
        rfm_features = self.segmentation.calculate_rfm_features(self.sample_data)
        
        # Check RFM columns exist
        expected_columns = ['customer_id', 'recency', 'frequency', 'monetary_total', 'monetary_avg']
        for col in expected_columns:
            self.assertIn(col, rfm_features.columns)
        
        # Check data quality
        self.assertEqual(len(rfm_features), 100)  # Should have 100 unique customers
        self.assertTrue(all(rfm_features['frequency'] > 0))  # All customers should have transactions
    
    def test_model_training(self):
        """Test customer segmentation training"""
        self.segmentation.train(data=self.sample_data)
        
        # Verify training
        self.assertTrue(self.segmentation.is_trained)
        self.assertIsNotNone(self.segmentation.segment_profiles)
    
    def test_segment_prediction(self):
        """Test customer segment prediction"""
        # Train model first
        self.segmentation.train(data=self.sample_data)
        
        # Predict segments for new data
        predictions = self.segmentation.predict_segment(self.sample_data.head(50))
        
        # Verify predictions
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)
        
        # Check prediction structure
        first_pred = predictions[0]
        required_keys = ['customer_id', 'cluster', 'segment_name']
        for key in required_keys:
            self.assertIn(key, first_pred)

class TestFraudDetector(unittest.TestCase):
    """Test cases for Fraud Detector model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fraud_detector = FraudDetector()
        
        # Create sample transaction data with some fraud cases
        self.sample_data = pd.DataFrame({
            'customer_id': np.random.randint(1, 100, 1000),
            'total_amount': np.random.lognormal(3, 1, 1000),
            'quantity': np.random.randint(1, 10, 1000),
            'unit_price': np.random.uniform(10, 500, 1000),
            'payment_method': np.random.choice(['cash', 'credit_card', 'debit_card'], 1000),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food'], 1000),
            'transaction_timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
            'store_location': np.random.choice(['Store_A', 'Store_B'], 1000)
        })
        
        # Add fraud labels
        self.sample_data['is_fraud'] = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    
    def test_initialization(self):
        """Test fraud detector initialization"""
        self.assertIsNotNone(self.fraud_detector.isolation_forest)
        self.assertIsNotNone(self.fraud_detector.classifier)
        self.assertFalse(self.fraud_detector.is_trained)
    
    def test_fraud_feature_creation(self):
        """Test fraud detection feature creation"""
        features = self.fraud_detector.create_fraud_features(self.sample_data)
        
        # Check that features are enhanced
        self.assertGreater(len(features.columns), len(self.sample_data.columns))
        
        # Check for specific fraud features
        expected_features = ['hour', 'is_weekend', 'is_round_amount']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
    
    def test_business_rules(self):
        """Test business rule application"""
        fraud_flags = self.fraud_detector.apply_business_rules(self.sample_data)
        
        # Should return boolean series
        self.assertIsInstance(fraud_flags, pd.Series)
        self.assertEqual(len(fraud_flags), len(self.sample_data))
        self.assertTrue(fraud_flags.dtype == bool)
    
    def test_model_training(self):
        """Test fraud detection training"""
        self.fraud_detector.train(data=self.sample_data)
        
        # Verify training
        self.assertTrue(self.fraud_detector.is_trained)
        self.assertIsNotNone(self.fraud_detector.feature_names)
    
    def test_fraud_detection(self):
        """Test fraud detection on new transactions"""
        # Train model first
        self.fraud_detector.train(data=self.sample_data)
        
        # Test fraud detection
        test_data = self.sample_data.head(10).copy()
        results = self.fraud_detector.detect_fraud(test_data)
        
        # Verify results structure
        self.assertIn('total_transactions', results)
        self.assertIn('transactions', results)
        self.assertEqual(results['total_transactions'], 10)
        self.assertEqual(len(results['transactions']), 10)
        
        # Check individual transaction results
        first_result = results['transactions'][0]
        required_keys = ['risk_level', 'overall_risk_score', 'explanations']
        for key in required_keys:
            self.assertIn(key, first_result)

class TestPOSInterface(unittest.TestCase):
    """Test cases for POS Interface"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock AI models
        self.sales_predictor = SalesPredictor()
        self.customer_segmentation = CustomerSegmentation()
        self.fraud_detector = FraudDetector()
        
        # Initialize POS interface
        self.pos_interface = POSInterface(
            sales_predictor=self.sales_predictor,
            customer_segmentation=self.customer_segmentation,
            fraud_detector=self.fraud_detector
        )
    
    def test_initialization(self):
        """Test POS interface initialization"""
        self.assertIsNotNone(self.pos_interface.inventory)
        self.assertIsInstance(self.pos_interface.session_stats, dict)
        self.assertEqual(self.pos_interface.session_stats['transactions_processed'], 0)
    
    def test_inventory_initialization(self):
        """Test inventory initialization"""
        inventory = self.pos_interface.inventory
        
        # Check inventory structure
        self.assertGreater(len(inventory), 0)
        
        # Check inventory item structure
        first_item = list(inventory.values())[0]
        required_keys = ['name', 'category', 'price', 'stock']
        for key in required_keys:
            self.assertIn(key, first_item)
    
    def test_transaction_creation(self):
        """Test transaction data structure"""
        # Create a sample transaction
        transaction = {
            'transaction_id': 'TXN_000001',
            'customer_id': 'CUST_00001',
            'product_id': 'ELEC001',
            'quantity': 2,
            'unit_price': 599.99,
            'total_amount': 1199.98,
            'payment_method': 'credit_card',
            'transaction_timestamp': datetime.now()
        }
        
        # Test transaction structure
        required_keys = ['transaction_id', 'customer_id', 'total_amount']
        for key in required_keys:
            self.assertIn(key, transaction)
        
        # Test data types
        self.assertIsInstance(transaction['total_amount'], (int, float))
        self.assertIsInstance(transaction['quantity'], int)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create sample data
        self.sample_transactions = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'] * 100,
            'transaction_date': pd.date_range('2024-01-01', periods=300),
            'total_amount': np.random.lognormal(3, 1, 300),
            'quantity': np.random.randint(1, 5, 300),
            'unit_price': np.random.uniform(10, 100, 300),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food'], 300),
            'payment_method': np.random.choice(['cash', 'credit_card', 'debit_card'], 300),
            'transaction_timestamp': pd.date_range('2024-01-01', periods=300, freq='H'),
            'store_location': 'Store_A'
        })
    
    def test_end_to_end_workflow(self):
        """Test complete AI workflow integration"""
        # Initialize models
        sales_predictor = SalesPredictor()
        customer_segmentation = CustomerSegmentation(n_clusters=3)
        fraud_detector = FraudDetector()
        
        # Train all models
        try:
            sales_predictor.train(data=self.sample_transactions)
            customer_segmentation.train(data=self.sample_transactions)
            fraud_detector.train(data=self.sample_transactions)
            
            # Verify all models are trained
            self.assertTrue(sales_predictor.is_trained)
            self.assertTrue(customer_segmentation.is_trained)
            self.assertTrue(fraud_detector.is_trained)
            
            # Test predictions
            sales_pred = sales_predictor.predict_sales(data=self.sample_transactions)
            segment_pred = customer_segmentation.predict_segment(self.sample_transactions)
            fraud_pred = fraud_detector.detect_fraud(self.sample_transactions.head(10))
            
            # Verify predictions
            self.assertIsNotNone(sales_pred)
            self.assertIsNotNone(segment_pred)
            self.assertIsNotNone(fraud_pred)
            
        except Exception as e:
            self.fail(f"End-to-end workflow failed: {str(e)}")
    
    def test_pos_system_integration(self):
        """Test POS system with trained models"""
        # Train models
        sales_predictor = SalesPredictor()
        customer_segmentation = CustomerSegmentation(n_clusters=3)
        fraud_detector = FraudDetector()
        
        sales_predictor.train(data=self.sample_transactions)
        customer_segmentation.train(data=self.sample_transactions)
        fraud_detector.train(data=self.sample_transactions)
        
        # Initialize POS system
        pos_system = POSInterface(
            sales_predictor=sales_predictor,
            customer_segmentation=customer_segmentation,
            fraud_detector=fraud_detector
        )
        
        # Test system components
        self.assertIsNotNone(pos_system.sales_predictor)
        self.assertIsNotNone(pos_system.customer_segmentation)
        self.assertIsNotNone(pos_system.fraud_detector)
        
        # Test fraud detection on sample transaction
        sample_transaction = {
            'customer_id': 'CUST_001',
            'total_amount': 1000.0,
            'quantity': 1,
            'unit_price': 1000.0,
            'payment_method': 'credit_card',
            'product_category': 'Electronics',
            'transaction_timestamp': datetime.now(),
            'store_location': 'Store_A'
        }
        
        fraud_result = pos_system._check_transaction_fraud(sample_transaction)
        self.assertIn('risk_level', fraud_result)
        self.assertIn('overall_risk_score', fraud_result)

def run_tests():
    """Run all unit tests"""
    print("🧪 Running Unit Tests for Horizon AI-Powered POS System")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestSalesPredictor,
        TestCustomerSegmentation,
        TestFraudDetector,
        TestPOSInterface,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🧪 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print("\n⚠️ ERRORS:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    if not result.failures and not result.errors:
        print("\n✅ All tests passed successfully!")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)