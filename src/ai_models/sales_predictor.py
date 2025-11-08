"""
Sales Predictor AI Model
========================

This module implements a machine learning model for predicting future sales
based on historical transaction data, seasonal patterns, and external factors.

The model uses Random Forest algorithm for robust prediction with feature importance insights.

Author: [Your Name]
Course: AI for Software Engineering  
Date: November 8, 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SalesPredictor:
    """
    AI model for predicting sales based on historical data and patterns
    
    This class implements a Random Forest-based sales prediction system
    that can forecast future sales volumes and revenue.
    """
    
    def __init__(self):
        """Initialize the Sales Predictor with default parameters"""
        
        # Model and preprocessing components
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Training data storage
        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.is_trained = False
        
        logger.info("Sales Predictor initialized")
    
    def prepare_features(self, data):
        """
        Prepare features for model training or prediction
        
        Args:
            data (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Processed features ready for ML model
        """
        logger.info("Preparing features for sales prediction...")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Convert date columns
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            
            # Extract temporal features
            df['year'] = df['transaction_date'].dt.year
            df['month'] = df['transaction_date'].dt.month
            df['day_of_week'] = df['transaction_date'].dt.dayofweek
            df['day_of_month'] = df['transaction_date'].dt.day
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['quarter'] = df['transaction_date'].dt.quarter
        
        # Product category encoding
        if 'product_category' in df.columns:
            if 'product_category' not in self.label_encoders:
                self.label_encoders['product_category'] = LabelEncoder()
                df['product_category_encoded'] = self.label_encoders['product_category'].fit_transform(df['product_category'])
            else:
                df['product_category_encoded'] = self.label_encoders['product_category'].transform(df['product_category'])
        
        # Customer segment encoding
        if 'customer_segment' in df.columns:
            if 'customer_segment' not in self.label_encoders:
                self.label_encoders['customer_segment'] = LabelEncoder()
                df['customer_segment_encoded'] = self.label_encoders['customer_segment'].fit_transform(df['customer_segment'])
            else:
                df['customer_segment_encoded'] = self.label_encoders['customer_segment'].transform(df['customer_segment'])
        
        # Aggregate features by date
        if 'transaction_date' in df.columns:
            # Daily aggregations
            daily_features = df.groupby('transaction_date').agg({
                'quantity': ['sum', 'mean', 'std'],
                'unit_price': ['mean', 'std'],
                'total_amount': ['sum', 'mean'],
                'product_category_encoded': 'nunique',
                'customer_id': 'nunique'
            }).fillna(0)
            
            # Flatten column names
            daily_features.columns = ['_'.join(col).strip() for col in daily_features.columns]
            
            # Add temporal features
            daily_features['year'] = daily_features.index.year
            daily_features['month'] = daily_features.index.month
            daily_features['day_of_week'] = daily_features.index.dayofweek
            daily_features['is_weekend'] = daily_features.index.dayofweek.isin([5, 6]).astype(int)
            daily_features['quarter'] = daily_features.index.quarter
            
            # Create lag features (previous days' sales)
            for lag in [1, 3, 7, 14]:
                daily_features[f'total_amount_sum_lag_{lag}'] = daily_features['total_amount_sum'].shift(lag)
            
            # Rolling averages
            for window in [3, 7, 14]:
                daily_features[f'total_amount_sum_ma_{window}'] = daily_features['total_amount_sum'].rolling(window=window).mean()
            
            # Fill NaN values
            daily_features.fillna(0, inplace=True)
            
            return daily_features.reset_index()
        
        return df
    
    def train(self, data_path=None, data=None, target_column='total_amount_sum'):
        """
        Train the sales prediction model
        
        Args:
            data_path (str): Path to training data CSV file
            data (pd.DataFrame): Training data (alternative to data_path)
            target_column (str): Name of target variable to predict
        """
        logger.info("Starting sales prediction model training...")
        
        try:
            # Load data
            if data_path:
                raw_data = pd.read_csv(data_path)
                logger.info(f"Loaded training data from {data_path}: {raw_data.shape}")
            elif data is not None:
                raw_data = data.copy()
                logger.info(f"Using provided training data: {raw_data.shape}")
            else:
                raise ValueError("Either data_path or data must be provided")
            
            # Prepare features
            processed_data = self.prepare_features(raw_data)
            
            # Separate features and target
            if target_column not in processed_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Define feature columns (exclude target and date columns)
            feature_columns = [col for col in processed_data.columns 
                             if col not in [target_column, 'transaction_date']]
            
            X = processed_data[feature_columns]
            y = processed_data[target_column]
            
            # Store feature names
            self.feature_names = feature_columns
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Hyperparameter tuning
            logger.info("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Use best model
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            # Train final model
            self.model.fit(X_train_scaled, y_train)
            
            # Store training data
            self.X_train = X_train_scaled
            self.y_train = y_train
            
            # Evaluate model
            train_pred = self.model.predict(X_train_scaled)
            val_pred = self.model.predict(X_val_scaled)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_r2 = r2_score(y_train, train_pred)
            
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            
            # Log results
            logger.info("Training Results:")
            logger.info(f"  Training MAE: {train_mae:.2f}")
            logger.info(f"  Training RMSE: {train_rmse:.2f}")
            logger.info(f"  Training R²: {train_r2:.3f}")
            logger.info(f"  Validation MAE: {val_mae:.2f}")
            logger.info(f"  Validation RMSE: {val_rmse:.2f}")
            logger.info(f"  Validation R²: {val_r2:.3f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            self.is_trained = True
            logger.info("Sales prediction model training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def predict_sales(self, data=None, days_ahead=7):
        """
        Predict future sales
        
        Args:
            data (pd.DataFrame): Current/recent data for prediction context
            days_ahead (int): Number of days to predict ahead
            
        Returns:
            dict: Dictionary containing predictions and confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Predicting sales for {days_ahead} days ahead...")
        
        try:
            # For demonstration, create synthetic recent data if none provided
            if data is None:
                # Create sample data for the last 30 days
                dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
                data = pd.DataFrame({
                    'transaction_date': dates,
                    'total_amount': np.random.normal(5000, 1000, 30),
                    'quantity': np.random.normal(100, 20, 30),
                    'unit_price': np.random.normal(50, 10, 30),
                    'product_category': np.random.choice(['Electronics', 'Clothing', 'Food'], 30),
                    'customer_segment': np.random.choice(['Regular', 'Premium', 'VIP'], 30)
                })
            
            # Prepare features
            processed_data = self.prepare_features(data)
            
            # Get features for prediction
            if self.feature_names:
                X_pred = processed_data[self.feature_names].iloc[-1:].values
            else:
                # Use last row of processed data
                feature_cols = [col for col in processed_data.columns 
                               if col not in ['transaction_date', 'total_amount_sum']]
                X_pred = processed_data[feature_cols].iloc[-1:].values
            
            # Scale features
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Make predictions for multiple days
            predictions = []
            for day in range(days_ahead):
                pred = self.model.predict(X_pred_scaled)[0]
                predictions.append(pred)
                
                # For next day prediction, you would normally update features
                # For now, we'll add some random variation
                X_pred_scaled = X_pred_scaled * (1 + np.random.normal(0, 0.05))
            
            # Calculate prediction intervals (using model's estimators)
            pred_intervals = []
            for pred in predictions:
                # Simple confidence interval based on training error
                margin = pred * 0.1  # 10% margin as example
                pred_intervals.append({
                    'prediction': pred,
                    'lower_bound': pred - margin,
                    'upper_bound': pred + margin
                })
            
            result = {
                'predictions': predictions,
                'prediction_intervals': pred_intervals,
                'total_predicted': sum(predictions),
                'average_daily': np.mean(predictions),
                'days_ahead': days_ahead
            }
            
            logger.info(f"Sales prediction completed: ${sum(predictions):.2f} total over {days_ahead} days")
            return result
            
        except Exception as e:
            logger.error(f"Error during sales prediction: {str(e)}")
            raise
    
    def get_feature_importance(self):
        """
        Get feature importance from trained model
        
        Returns:
            pd.DataFrame: Feature importance rankings
        """
        if not self.is_trained or self.feature_names is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, file_path):
        """
        Save trained model to disk
        
        Args:
            file_path (str): Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Sales prediction model saved to {file_path}")
    
    def load_model(self, file_path):
        """
        Load trained model from disk
        
        Args:
            file_path (str): Path to saved model file
        """
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Sales prediction model loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise