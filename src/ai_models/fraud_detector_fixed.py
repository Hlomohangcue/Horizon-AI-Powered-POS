"""
Fixed Fraud Detector that handles string columns properly
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FraudDetectorFixed:
    """Fixed Fraud Detection System that properly handles data types"""
    
    def __init__(self, contamination=0.1):
        """Initialize the fraud detector"""
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        logger.info("Fixed Fraud Detector initialized")
    
    def prepare_numeric_features(self, data):
        """Prepare only numeric features for fraud detection"""
        df = data.copy()
        features = pd.DataFrame()
        
        # Only include numeric columns and properly encoded categorical columns
        numeric_cols = ['quantity', 'unit_price', 'total_amount', 'customer_id']
        
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, handling any string values
                features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Add time-based features if timestamp exists
        if 'transaction_timestamp' in df.columns:
            df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
            features['hour'] = df['transaction_timestamp'].dt.hour
            features['day_of_week'] = df['transaction_timestamp'].dt.dayofweek
            features['month'] = df['transaction_timestamp'].dt.month
        elif 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            features['hour'] = 12  # Default hour
            features['day_of_week'] = df['transaction_date'].dt.dayofweek
            features['month'] = df['transaction_date'].dt.month
        else:
            # Default time features
            features['hour'] = 12
            features['day_of_week'] = 1
            features['month'] = 1
        
        # Encode categorical columns properly
        if 'payment_method' in df.columns:
            if 'payment_method' not in self.label_encoders:
                self.label_encoders['payment_method'] = LabelEncoder()
                features['payment_method_encoded'] = self.label_encoders['payment_method'].fit_transform(df['payment_method'].fillna('unknown'))
            else:
                # Handle unseen categories
                try:
                    features['payment_method_encoded'] = self.label_encoders['payment_method'].transform(df['payment_method'].fillna('unknown'))
                except ValueError:
                    features['payment_method_encoded'] = 0
        else:
            features['payment_method_encoded'] = 0
        
        # Category encoding
        if 'category' in df.columns:
            if 'category' not in self.label_encoders:
                self.label_encoders['category'] = LabelEncoder()
                features['category_encoded'] = self.label_encoders['category'].fit_transform(df['category'].fillna('unknown'))
            else:
                try:
                    features['category_encoded'] = self.label_encoders['category'].transform(df['category'].fillna('unknown'))
                except ValueError:
                    features['category_encoded'] = 0
        elif 'product_category' in df.columns:
            if 'product_category' not in self.label_encoders:
                self.label_encoders['product_category'] = LabelEncoder()
                features['category_encoded'] = self.label_encoders['product_category'].fit_transform(df['product_category'].fillna('unknown'))
            else:
                try:
                    features['category_encoded'] = self.label_encoders['product_category'].transform(df['product_category'].fillna('unknown'))
                except ValueError:
                    features['category_encoded'] = 0
        else:
            features['category_encoded'] = 0
        
        # Create derived fraud detection features
        if 'total_amount' in features.columns and 'quantity' in features.columns:
            features['amount_per_unit'] = features['total_amount'] / (features['quantity'] + 1)  # Avoid division by zero
            features['is_high_amount'] = (features['total_amount'] > features['total_amount'].quantile(0.9)).astype(int)
            features['is_round_amount'] = (features['total_amount'] % 10 == 0).astype(int)
        
        # Ensure all features are numeric
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        
        self.feature_columns = features.columns.tolist()
        logger.info(f"Prepared {len(features.columns)} numeric features: {features.columns.tolist()}")
        
        return features
    
    def train(self, data=None, data_path=None):
        """Train the fraud detection model"""
        logger.info("Training fraud detection model...")
        
        try:
            # Load data
            if data_path:
                df = pd.read_csv(data_path)
            else:
                df = data.copy()
            
            # Prepare features
            X = self.prepare_numeric_features(df)
            
            # Create synthetic fraud labels for training (since we don't have real fraud data)
            # In practice, you would have actual fraud labels
            np.random.seed(42)
            y = np.random.choice([0, 1], size=len(X), p=[0.95, 0.05])  # 5% fraud rate
            
            # Train isolation forest for anomaly detection
            self.isolation_forest.fit(X)
            
            # Scale features for classifier
            X_scaled = self.scaler.fit_transform(X)
            
            # Train classifier
            self.classifier.fit(X_scaled, y)
            
            self.is_trained = True
            logger.info("✅ Fraud detection model training completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training fraud detector: {e}")
            return False
    
    def detect_fraud(self, data):
        """Detect fraud in transaction data"""
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return np.zeros(len(data))
        
        try:
            # Prepare features
            X = self.prepare_numeric_features(data)
            
            # Ensure we have the same features as training
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            
            X = X[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from both models
            anomaly_scores = self.isolation_forest.decision_function(X_scaled)
            fraud_probs = self.classifier.predict_proba(X_scaled)[:, 1]
            
            # Combine predictions (weighted average)
            combined_scores = 0.6 * fraud_probs + 0.4 * (1 - (anomaly_scores + 1) / 2)
            
            # Return binary predictions (threshold = 0.5)
            return (combined_scores > 0.5).astype(int)
            
        except Exception as e:
            logger.error(f"Error detecting fraud: {e}")
            return np.zeros(len(data))
    
    def save_model(self, filepath):
        """Save the trained model"""
        try:
            model_data = {
                'isolation_forest': self.isolation_forest,
                'classifier': self.classifier,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.isolation_forest = model_data['isolation_forest']
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False