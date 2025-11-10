# 🏪 Horizon AI-Powered POS System
*AI for Software Engineering - Week 5 Assignment*

## 📋 Assignment Overview
**Student:** [Your Name]  
**Course:** AI for Software Engineering  
**Assignment:** Week 5 - AI-Powered Business Application  
**Date:** November 10, 2025  
**Currency:** Lesotho Maloti (LSL) 🇱🇸

## 🎯 Project Description
A comprehensive Point of Sale (POS) system designed for Horizon Enterprise in Lesotho, featuring AI-powered analytics, inventory management, and sales processing with complete Streamlit web interface.

## 🌐 Live Demo
**Try the application now:** https://hlomohangcue-horizon-ai-powered-pos-streamlit-app-cxyme1.streamlit.app/

*Experience the full POS system with AI-powered features, inventory management, and sales analytics - all running with Lesotho Maloti currency!*

## ✨ Key Features

### 🏪 Core POS Functionality
- **Sales Processing:** Complete transaction management with change calculation
- **Inventory Management:** Real-time stock tracking and management
- **Receipt Generation:** Professional digital receipts with Maloti currency
- **Multi-Role Interface:** Separate dashboards for sales assistants and managers

### 🤖 AI-Powered Intelligence
- **Sales Prediction:** Machine learning-based revenue forecasting
- **Customer Segmentation:** RFM analysis for customer insights
- **Fraud Detection:** Real-time transaction risk assessment
- **Business Analytics:** Comprehensive sales and inventory analytics

### 💰 Lesotho-Specific Features
- **Currency:** All pricing in Lesotho Maloti (M XXX.XX format)
- **Local Context:** Designed for Lesotho business environment
- **Change Breakdown:** Maloti bill denominations (M 20, M 10, M 5, M 1)

## 🏗️ System Architecture

```
horizon-ai-pos/
├── streamlit_app.py              # Main web application
├── src/
│   ├── ai_models/                # AI/ML Components
│   │   ├── sales_predictor.py    # Sales forecasting model
│   │   ├── customer_segmentation.py # RFM customer analysis
│   │   └── fraud_detector_fixed.py  # Fraud detection system
│   └── pos_system/               # Core POS Logic
│       ├── pos_interface.py      # Terminal-based interface
│       └── enhanced_pos_interface.py # Enhanced features
├── data/                         # CSV Data Storage
│   ├── inventory.csv            # Product inventory
│   ├── transactions.csv         # Sales transactions
│   └── customers.csv            # Customer database
├── tests/                       # Test Suite
├── requirements.txt             # Python dependencies
└── README.md                   # This documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11+
- Streamlit 1.28+
- Pandas, NumPy, Scikit-learn, Plotly

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/Hlomohangcue/Horizon-AI-Powered-POS.git
cd Horizon-AI-Powered-POS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run streamlit_app.py
```

### Access the Application
- **🌐 Live Demo:** https://hlomohangcue-horizon-ai-powered-pos-streamlit-app-cxyme1.streamlit.app/
- **💻 Local Development:** http://localhost:8501
- **Features:** Sales, Inventory, Analytics, AI Insights

## Usage

### Training Models
```python
from src.ai_models.sales_predictor import SalesPredictor

# Initialize and train the sales prediction model
predictor = SalesPredictor()
predictor.train('data/sales_history.csv')
predictor.save_model('models/sales_predictor.pkl')
```

### Making Predictions
```python
# Predict next week's sales
predictions = predictor.predict_sales(days_ahead=7)
print(f"Predicted sales: {predictions}")
```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions and support, please contact the development team at support@horizonenterprise.com