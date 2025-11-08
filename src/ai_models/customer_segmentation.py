"""
Customer Segmentation AI Model
==============================

This module implements machine learning algorithms for customer segmentation
based on purchasing behavior, demographics, and transaction patterns.

Uses K-Means clustering and RFM (Recency, Frequency, Monetary) analysis.

Author: [Your Name]
Course: AI for Software Engineering
Date: November 8, 2025
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CustomerSegmentation:
    """
    AI model for customer segmentation using RFM analysis and clustering
    
    This class implements customer segmentation to identify different
    customer groups for targeted marketing and personalized recommendations.
    """
    
    def __init__(self, n_clusters=5):
        """
        Initialize Customer Segmentation model
        
        Args:
            n_clusters (int): Number of customer segments to create
        """
        self.n_clusters = n_clusters
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)  # For visualization
        
        self.is_trained = False
        self.feature_names = None
        self.segment_profiles = None
        
        logger.info(f"Customer Segmentation initialized with {n_clusters} clusters")
    
    def calculate_rfm_features(self, data, customer_id_col='customer_id', 
                              date_col='transaction_date', amount_col='total_amount'):
        """
        Calculate RFM (Recency, Frequency, Monetary) features for customers
        
        Args:
            data (pd.DataFrame): Transaction data
            customer_id_col (str): Column name for customer ID
            date_col (str): Column name for transaction date
            amount_col (str): Column name for transaction amount
            
        Returns:
            pd.DataFrame: RFM features for each customer
        """
        logger.info("Calculating RFM features...")
        
        # Make copy and ensure date column is datetime
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Calculate reference date (most recent transaction date)
        reference_date = df[date_col].max()
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_id_col).agg({
            date_col: lambda x: (reference_date - x.max()).days,  # Recency
            amount_col: ['count', 'sum', 'mean']  # Frequency and Monetary
        }).round(2)
        
        # Flatten column names
        rfm.columns = ['recency', 'frequency', 'monetary_total', 'monetary_avg']
        
        # Calculate additional behavioral features
        df_customer_behavior = df.groupby(customer_id_col).agg({
            'quantity': ['sum', 'mean', 'std'],
            'unit_price': ['mean', 'std'],
            date_col: ['count', 'nunique'],  # Transaction count and unique days
            'product_category': 'nunique' if 'product_category' in df.columns else lambda x: 0
        }).round(2)
        
        # Flatten column names
        df_customer_behavior.columns = [
            'total_quantity', 'avg_quantity', 'std_quantity',
            'avg_unit_price', 'std_unit_price',
            'transaction_count', 'active_days',
            'product_categories'
        ]
        
        # Combine RFM with behavioral features
        customer_features = rfm.join(df_customer_behavior, how='inner')
        
        # Fill NaN values
        customer_features.fillna(0, inplace=True)
        
        # Calculate derived features
        customer_features['avg_days_between_purchases'] = np.where(
            customer_features['active_days'] > 1,
            customer_features['recency'] / customer_features['active_days'],
            customer_features['recency']
        )
        
        customer_features['customer_lifetime_value'] = (
            customer_features['monetary_total'] / customer_features['recency'].replace(0, 1)
        )
        
        # Customer loyalty score (combination of frequency and recency)
        customer_features['loyalty_score'] = (
            (1 / (customer_features['recency'] + 1)) * 
            customer_features['frequency'] * 
            customer_features['monetary_avg']
        )
        
        logger.info(f"RFM features calculated for {len(customer_features)} customers")
        return customer_features.reset_index()
    
    def train(self, data_path=None, data=None, customer_id_col='customer_id'):
        """
        Train the customer segmentation model
        
        Args:
            data_path (str): Path to training data CSV file
            data (pd.DataFrame): Training data (alternative to data_path)
            customer_id_col (str): Column name for customer ID
        """
        logger.info("Starting customer segmentation model training...")
        
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
            
            # Calculate RFM features
            customer_features = self.calculate_rfm_features(raw_data, customer_id_col)
            
            # Prepare features for clustering (exclude customer_id)
            feature_columns = [col for col in customer_features.columns if col != customer_id_col]
            X = customer_features[feature_columns]
            
            # Store feature names
            self.feature_names = feature_columns
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Determine optimal number of clusters using elbow method
            logger.info("Determining optimal number of clusters...")
            inertias = []
            silhouette_scores = []
            k_range = range(2, min(11, len(customer_features) // 2))
            
            for k in k_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_temp.fit(X_scaled)
                inertias.append(kmeans_temp.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))
            
            # Use the k with highest silhouette score
            optimal_k = k_range[np.argmax(silhouette_scores)]
            logger.info(f"Optimal number of clusters: {optimal_k}")
            
            # Update model with optimal k if different
            if optimal_k != self.n_clusters:
                self.n_clusters = optimal_k
                self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            
            # Train final model
            self.kmeans_model.fit(X_scaled)
            
            # Get cluster labels
            cluster_labels = self.kmeans_model.labels_
            customer_features['cluster'] = cluster_labels
            
            # Calculate cluster profiles
            self.segment_profiles = customer_features.groupby('cluster').agg({
                'recency': ['mean', 'std'],
                'frequency': ['mean', 'std'],
                'monetary_total': ['mean', 'std'],
                'monetary_avg': ['mean', 'std'],
                'loyalty_score': ['mean', 'std'],
                'customer_lifetime_value': ['mean', 'std'],
                customer_id_col: 'count'  # Size of each cluster
            }).round(2)
            
            # Flatten column names
            self.segment_profiles.columns = [
                'recency_mean', 'recency_std',
                'frequency_mean', 'frequency_std',
                'monetary_total_mean', 'monetary_total_std',
                'monetary_avg_mean', 'monetary_avg_std',
                'loyalty_score_mean', 'loyalty_score_std',
                'clv_mean', 'clv_std',
                'cluster_size'
            ]
            
            # Assign segment names based on characteristics
            segment_names = self._assign_segment_names()
            self.segment_profiles['segment_name'] = segment_names
            
            # Calculate silhouette score for final model
            final_silhouette = silhouette_score(X_scaled, cluster_labels)
            
            # Log results
            logger.info("Customer Segmentation Training Results:")
            logger.info(f"  Number of customers: {len(customer_features)}")
            logger.info(f"  Number of clusters: {self.n_clusters}")
            logger.info(f"  Silhouette Score: {final_silhouette:.3f}")
            logger.info(f"  Inertia: {self.kmeans_model.inertia_:.2f}")
            
            # Log segment profiles
            logger.info("Customer Segment Profiles:")
            for idx, row in self.segment_profiles.iterrows():
                logger.info(f"  Cluster {idx} ({row['segment_name']}): "
                           f"{row['cluster_size']} customers, "
                           f"Avg CLV: ${row['clv_mean']:.2f}")
            
            # Store customer features for reference
            self.customer_features = customer_features
            
            self.is_trained = True
            logger.info("Customer segmentation model training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def _assign_segment_names(self):
        """
        Assign meaningful names to customer segments based on their characteristics
        
        Returns:
            list: List of segment names
        """
        segment_names = []
        
        for idx, row in self.segment_profiles.iterrows():
            # Define segments based on RFM characteristics
            recency = row['recency_mean']
            frequency = row['frequency_mean']
            monetary = row['monetary_avg_mean']
            loyalty = row['loyalty_score_mean']
            
            if loyalty > self.segment_profiles['loyalty_score_mean'].quantile(0.8):
                if monetary > self.segment_profiles['monetary_avg_mean'].quantile(0.7):
                    name = "VIP Champions"
                else:
                    name = "Loyal Customers"
            elif recency < self.segment_profiles['recency_mean'].quantile(0.3):
                if frequency > self.segment_profiles['frequency_mean'].median():
                    name = "Active Regulars"
                else:
                    name = "New Customers"
            elif recency > self.segment_profiles['recency_mean'].quantile(0.7):
                if monetary > self.segment_profiles['monetary_avg_mean'].median():
                    name = "At-Risk High Value"
                else:
                    name = "At-Risk Low Value"
            else:
                name = "Potential Loyalists"
            
            segment_names.append(name)
        
        return segment_names
    
    def predict_segment(self, customer_data):
        """
        Predict customer segment for new customers
        
        Args:
            customer_data (pd.DataFrame): Customer transaction data
            
        Returns:
            dict: Customer segments and characteristics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info("Predicting customer segments...")
        
        try:
            # Calculate RFM features for new customers
            customer_features = self.calculate_rfm_features(customer_data)
            
            # Prepare features (exclude customer_id)
            X = customer_features[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict clusters
            cluster_labels = self.kmeans_model.predict(X_scaled)
            
            # Create results
            results = []
            for idx, customer_id in enumerate(customer_features['customer_id']):
                cluster = cluster_labels[idx]
                segment_info = self.segment_profiles.loc[cluster]
                
                results.append({
                    'customer_id': customer_id,
                    'cluster': cluster,
                    'segment_name': segment_info['segment_name'],
                    'cluster_size': segment_info['cluster_size'],
                    'predicted_clv': segment_info['clv_mean'],
                    'loyalty_level': self._get_loyalty_level(segment_info['loyalty_score_mean'])
                })
            
            logger.info(f"Segment prediction completed for {len(results)} customers")
            return results
            
        except Exception as e:
            logger.error(f"Error during segment prediction: {str(e)}")
            raise
    
    def _get_loyalty_level(self, loyalty_score):
        """
        Convert loyalty score to categorical level
        
        Args:
            loyalty_score (float): Numerical loyalty score
            
        Returns:
            str: Loyalty level category
        """
        if loyalty_score > 1000:
            return "Very High"
        elif loyalty_score > 500:
            return "High"
        elif loyalty_score > 100:
            return "Medium"
        elif loyalty_score > 10:
            return "Low"
        else:
            return "Very Low"
    
    def get_segment_recommendations(self, segment_name):
        """
        Get marketing recommendations for specific customer segment
        
        Args:
            segment_name (str): Name of customer segment
            
        Returns:
            dict: Marketing recommendations and strategies
        """
        recommendations = {
            "VIP Champions": {
                "strategy": "Retain and reward loyalty",
                "actions": [
                    "Exclusive VIP programs and early access to new products",
                    "Personal account managers and premium customer service",
                    "Invitation-only events and experiences",
                    "High-value loyalty rewards and cashback programs"
                ],
                "communication": "Personalized, premium channels"
            },
            "Loyal Customers": {
                "strategy": "Increase spend and referrals",
                "actions": [
                    "Upselling and cross-selling campaigns",
                    "Referral reward programs",
                    "Product bundling offers",
                    "Loyalty tier upgrade incentives"
                ],
                "communication": "Regular, value-focused messaging"
            },
            "Active Regulars": {
                "strategy": "Build loyalty and increase frequency",
                "actions": [
                    "Frequency-based reward programs",
                    "Personalized product recommendations",
                    "Time-sensitive offers to encourage regular visits",
                    "Educational content about product benefits"
                ],
                "communication": "Consistent, engagement-focused"
            },
            "New Customers": {
                "strategy": "Onboard and convert to regular customers",
                "actions": [
                    "Welcome series and onboarding campaigns",
                    "First-time buyer discounts and incentives",
                    "Product education and tutorials",
                    "Early engagement rewards"
                ],
                "communication": "Educational, welcoming tone"
            },
            "At-Risk High Value": {
                "strategy": "Win back and re-engage",
                "actions": [
                    "Personalized win-back campaigns",
                    "Special offers and discounts",
                    "Survey to understand reasons for disengagement",
                    "Exclusive comeback offers"
                ],
                "communication": "Urgent, value-proposition focused"
            },
            "At-Risk Low Value": {
                "strategy": "Re-engage with cost-effective methods",
                "actions": [
                    "Automated email campaigns with special offers",
                    "Social media remarketing",
                    "Simple surveys for feedback",
                    "Low-cost acquisition offers"
                ],
                "communication": "Automated, broad-appeal messaging"
            },
            "Potential Loyalists": {
                "strategy": "Nurture towards loyalty",
                "actions": [
                    "Gradual loyalty program introduction",
                    "Consistent value delivery",
                    "Engagement tracking and optimization",
                    "Targeted promotions based on preferences"
                ],
                "communication": "Nurturing, consistent engagement"
            }
        }
        
        return recommendations.get(segment_name, {
            "strategy": "General engagement",
            "actions": ["Standard marketing campaigns", "Regular promotions"],
            "communication": "Standard marketing channels"
        })
    
    def get_cluster_summary(self):
        """
        Get summary of all customer segments
        
        Returns:
            pd.DataFrame: Summary of all segments with key metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting cluster summary")
        
        return self.segment_profiles
    
    def save_model(self, file_path):
        """
        Save trained model to disk
        
        Args:
            file_path (str): Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'kmeans_model': self.kmeans_model,
            'scaler': self.scaler,
            'pca': self.pca,
            'n_clusters': self.n_clusters,
            'feature_names': self.feature_names,
            'segment_profiles': self.segment_profiles,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Customer segmentation model saved to {file_path}")
    
    def load_model(self, file_path):
        """
        Load trained model from disk
        
        Args:
            file_path (str): Path to saved model file
        """
        try:
            model_data = joblib.load(file_path)
            
            self.kmeans_model = model_data['kmeans_model']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.n_clusters = model_data['n_clusters']
            self.feature_names = model_data['feature_names']
            self.segment_profiles = model_data['segment_profiles']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Customer segmentation model loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise