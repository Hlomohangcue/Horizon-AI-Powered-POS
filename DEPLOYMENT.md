# 🚀 Horizon AI POS - Deployment Guide

Complete deployment guide for the Horizon AI-Powered POS System.

## 📋 Prerequisites

- Python 3.8+
- Git
- Internet connection for package installation

## 🏃‍♂️ Quick Start

### 1. Install Dependencies

```bash
# Install Streamlit
pip install streamlit

# Install all requirements
pip install -r requirements.txt
```

### 2. Setup and Run

```bash
# Setup the system
python enhanced_setup.py

# Start the web application
streamlit run streamlit_app.py
```

### 3. Access the Application

Open your browser and go to: `http://localhost:8501`

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `streamlit_app.py` as the main file
   - Click "Deploy"

3. **Your app will be live at**: `https://your-app-name.streamlit.app`

### Option 2: Heroku Deployment

1. **Install Heroku CLI**
2. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Docker Deployment

```bash
# Build image
docker build -t horizon-ai-pos .

# Run container
docker run -p 8501:8501 horizon-ai-pos
```

## 🔧 Configuration

### Environment Variables

Create a `.streamlit/secrets.toml` file for production:

```toml
[general]
app_name = "Horizon AI POS"
debug_mode = false

[database]
# Add database configurations if needed
```

### Performance Optimization

For production deployment:

1. **Caching**: The app uses Streamlit's caching for better performance
2. **Memory**: Recommended 1GB+ RAM for optimal performance
3. **Storage**: Ensure sufficient disk space for data and models

## 📊 Monitoring

### Application Health

The app includes built-in health monitoring:
- Response time tracking
- Error logging
- Usage analytics

### Logs

Check logs in the `logs/` directory:
- `application.log`: General application logs
- `model_performance.log`: AI model performance metrics

## 🔒 Security

### Production Security Checklist

- [ ] Use HTTPS in production
- [ ] Implement proper authentication
- [ ] Secure data storage
- [ ] Regular security updates
- [ ] Monitor for suspicious activity

## 🆘 Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

2. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Model Training Errors**:
   ```bash
   python enhanced_setup.py --force-retrain
   ```

### Support

For deployment support:
- Check the main README.md
- Review error logs in `logs/` directory
- Ensure all dependencies are installed

## 📈 Scaling

### For High Traffic

1. **Use Cloud Services**: AWS, GCP, or Azure
2. **Load Balancing**: Multiple app instances
3. **Database**: Move to external database
4. **Caching**: Redis for improved performance

### Multi-Store Deployment

1. **Database Separation**: Separate data per store
2. **Multi-tenancy**: Store-specific configurations
3. **Centralized Analytics**: Combined reporting dashboard

---

🎉 **Your Horizon AI POS System is now ready for deployment!**