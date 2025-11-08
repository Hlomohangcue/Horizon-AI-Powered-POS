"""
Enhanced Horizon AI POS System - Main Application
===============================================

This is the main entry point for the enhanced Horizon AI-powered POS system
with comprehensive inventory management, sales workflow, and AI insights.

Features:
- Manager Dashboard with full system control
- Sales Assistant Mode with streamlined workflow
- Real-time inventory management
- AI-powered insights and recommendations
- Comprehensive reporting and analytics

Author: Horizon Enterprise Team
Course: AI for Software Engineering
Date: November 8, 2025
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/horizon_pos_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import AI models
try:
    from ai_models.sales_predictor import SalesPredictor
    from ai_models.customer_segmentation import CustomerSegmentation
    from ai_models.fraud_detector_fixed import FraudDetectorFixed as FraudDetector
    AI_MODELS_AVAILABLE = True
    logger.info("AI models imported successfully")
except ImportError as e:
    logger.warning(f"AI models not available: {e}")
    AI_MODELS_AVAILABLE = False

# Import enhanced POS interface
from pos_system.enhanced_pos_interface import EnhancedPOSInterface

class HorizonEnhancedPOSSystem:
    """
    Enhanced Horizon AI POS System with complete business workflow
    """
    
    def __init__(self):
        """Initialize the enhanced POS system"""
        logger.info("Initializing Horizon Enhanced POS System...")
        
        # Initialize AI models if available
        self.ai_models = None
        if AI_MODELS_AVAILABLE:
            self.ai_models = self.load_ai_models()
        
        # Initialize enhanced POS interface
        self.pos_interface = EnhancedPOSInterface(ai_models=self.ai_models)
        
        logger.info("Horizon Enhanced POS System initialized successfully")
    
    def load_ai_models(self):
        """Load trained AI models"""
        logger.info("Loading AI models...")
        
        models = type('AIModels', (), {})()
        
        try:
            # Load Sales Predictor
            models.sales_predictor = SalesPredictor()
            if os.path.exists('models/sales_predictor.pkl'):
                models.sales_predictor.load_model('models/sales_predictor.pkl')
                logger.info("Sales Predictor loaded")
            else:
                logger.warning("Sales Predictor model not found")
            
            # Load Customer Segmentation
            models.customer_segmentation = CustomerSegmentation()
            if os.path.exists('models/customer_segmentation.pkl'):
                models.customer_segmentation.load_model('models/customer_segmentation.pkl')
                logger.info("Customer Segmentation loaded")
            else:
                logger.warning("Customer Segmentation model not found")
            
            # Load Fraud Detector
            models.fraud_detector = FraudDetector()
            if os.path.exists('models/fraud_detector.pkl'):
                models.fraud_detector.load_model('models/fraud_detector.pkl')
                logger.info("Fraud Detector loaded")
            else:
                logger.warning("Fraud Detector model not found")
            
            return models
            
        except Exception as e:
            logger.error(f"Error loading AI models: {e}")
            return None
    
    def check_system_status(self):
        """Check system components status"""
        print("\n🔍 SYSTEM STATUS CHECK")
        print("-" * 25)
        
        # Check data directories
        directories = ['data', 'models', 'logs', 'reports']
        for directory in directories:
            if os.path.exists(directory):
                print(f"✅ {directory}/ directory exists")
            else:
                print(f"⚠️ {directory}/ directory missing (will be created)")
                os.makedirs(directory, exist_ok=True)
        
        # Check AI models
        if self.ai_models:
            print("✅ AI models loaded and ready")
        else:
            print("⚠️ AI models not available - run training first")
        
        # Check inventory system
        try:
            inventory = self.pos_interface.inventory_manager.load_inventory()
            print(f"✅ Inventory system ready ({len(inventory)} products)")
        except Exception as e:
            print(f"❌ Inventory system error: {e}")
        
        print("\n🚀 System ready for operation!")
    
    def start(self):
        """Start the enhanced POS system"""
        try:
            # Show system banner
            self.show_system_banner()
            
            # Check system status
            self.check_system_status()
            
            # Start the enhanced interface
            self.pos_interface.start_system()
            
        except KeyboardInterrupt:
            print("\n\n👋 System shutdown requested by user")
            logger.info("System shutdown by user interrupt")
        except Exception as e:
            print(f"\n❌ System error: {e}")
            logger.error(f"System error: {e}")
        finally:
            logger.info("Horizon Enhanced POS System shutting down")
    
    def show_system_banner(self):
        """Display system banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║    🏢 HORIZON ENTERPRISE - AI POWERED POS SYSTEM (ENHANCED)                     ║
║                                                                                  ║
║    ✨ Features:                                                                  ║
║    • 👔 Manager Dashboard - Complete inventory & business control               ║
║    • 🛒 Sales Assistant Mode - Streamlined daily operations                     ║
║    • 📦 Real-time Inventory Management - Stock tracking & alerts                ║
║    • 🤖 AI-Powered Insights - Sales prediction & fraud detection               ║
║    • 📊 Comprehensive Analytics - Reports & business intelligence               ║
║    • 💾 Persistent Data Storage - CSV-based data management                     ║
║                                                                                  ║
║    🎓 AI for Software Engineering Course Project                                ║
║    📅 November 8, 2025                                                          ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)

def main():
    """Main application entry point"""
    try:
        # Initialize and start the enhanced POS system
        pos_system = HorizonEnhancedPOSSystem()
        pos_system.start()
        
    except Exception as e:
        print(f"❌ Failed to start system: {e}")
        logger.error(f"Failed to start system: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)