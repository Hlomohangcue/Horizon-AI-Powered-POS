"""
Horizon AI-Powered POS System - Main Application
=============================================

This is the main entry point for the Horizon AI-Powered POS System.
It initializes all AI models and starts the POS interface.

Author: [Your Name]
Course: AI for Software Engineering
Date: November 8, 2025
"""

import sys
import os
import logging
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pos_system.pos_interface import POSInterface
from src.ai_models.sales_predictor import SalesPredictor
from src.ai_models.customer_segmentation import CustomerSegmentation
from src.ai_models.fraud_detector_fixed import FraudDetectorFixed

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pos_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class HorizonPOSSystem:
    """
    Main class for the Horizon AI-Powered POS System
    
    This class orchestrates all the AI components and provides
    a unified interface for the Point of Sale operations.
    """
    
    def __init__(self):
        """Initialize the POS system with all AI components"""
        self.sales_predictor = None
        self.customer_segmentation = None
        self.fraud_detector = None
        self.pos_interface = None
        
        logger.info("Initializing Horizon AI-Powered POS System...")
        
    def initialize_ai_models(self):
        """
        Initialize and load all AI models
        
        This method loads pre-trained models or initializes new ones
        if no trained models are available.
        """
        try:
            # Initialize Sales Predictor
            logger.info("Loading Sales Prediction Model...")
            self.sales_predictor = SalesPredictor()
            if os.path.exists('models/sales_predictor.pkl'):
                self.sales_predictor.load_model('models/sales_predictor.pkl')
            else:
                logger.warning("No trained sales predictor found. Will need training.")
            
            # Initialize Customer Segmentation
            logger.info("Loading Customer Segmentation Model...")
            self.customer_segmentation = CustomerSegmentation()
            if os.path.exists('models/customer_segmentation.pkl'):
                self.customer_segmentation.load_model('models/customer_segmentation.pkl')
            else:
                logger.warning("No trained customer segmentation found. Will need training.")
            
            # Initialize Fraud Detector
            logger.info("Loading Fraud Detection Model...")
            self.fraud_detector = FraudDetectorFixed()
            if os.path.exists('models/fraud_detector.pkl'):
                self.fraud_detector.load_model('models/fraud_detector.pkl')
            else:
                logger.warning("No trained fraud detector found. Will need training.")
                
            logger.info("All AI models initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {str(e)}")
            raise
    
    def start_pos_system(self):
        """
        Start the main POS interface
        
        This method launches the user interface and begins
        accepting transactions and providing AI-powered insights.
        """
        try:
            logger.info("Starting POS Interface...")
            
            # Initialize POS Interface with AI models
            self.pos_interface = POSInterface(
                sales_predictor=self.sales_predictor,
                customer_segmentation=self.customer_segmentation,
                fraud_detector=self.fraud_detector
            )
            
            # Start the interface
            self.pos_interface.start()
            
        except Exception as e:
            logger.error(f"Error starting POS system: {str(e)}")
            raise
    
    def run(self):
        """
        Main execution method
        
        This method runs the complete POS system initialization
        and startup sequence.
        """
        try:
            print("="*60)
            print("🏪 HORIZON AI-POWERED POS SYSTEM")
            print("="*60)
            print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Initialize AI models
            self.initialize_ai_models()
            
            # Start POS system
            self.start_pos_system()
            
        except KeyboardInterrupt:
            logger.info("System shutdown requested by user")
            print("\n👋 Thank you for using Horizon AI-Powered POS System!")
            
        except Exception as e:
            logger.error(f"Critical error in main system: {str(e)}")
            print(f"❌ System error: {str(e)}")
            sys.exit(1)

def main():
    """Main entry point for the application"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize and run the POS system
    pos_system = HorizonPOSSystem()
    pos_system.run()

if __name__ == "__main__":
    main()