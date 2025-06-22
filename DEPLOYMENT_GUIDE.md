# 🚀 Food Detection App Deployment Guide

This guide will help you deploy your food detection application to free hosting platforms.

## 📋 Prerequisites

1. **GitHub Account**: For version control and deployment
2. **Model File**: Ensure your YOLO model is in the repository
3. **Python Knowledge**: Basic understanding of Python deployment

## 🎯 Deployment Options

### Option 1: Render (Recommended - Free Tier)

**Render** offers a generous free tier with automatic deployments from GitHub.

#### Step 1: Prepare Your Repository

1. **Create a GitHub repository** and push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Food Detection App"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/food-detection-app.git
   git push -u origin main
   ```

2. **Ensure these files are in your repository**:
   ```
   ├── food_detection_app_cloud.py    # Cloud-optimized app
   ├── requirements.txt               # Dependencies
   ├── render.yaml                    # Render configuration
   ├── templates/
   │   └── index.html                # Web interface
   └── runs/detect/yolov8n_food101/weights/
       └── best.pt                   # Your trained model
   ```

#### Step 2: Deploy on Render

1. **Go to [render.com](https://render.com)** and sign up
2. **Click "New +"** → **"Web Service"**
3. **Connect your GitHub repository**
4. **Configure the service**:
   - **Name**: `food-detection-ai`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn food_detection_app_cloud:app --bind 0.0.0.0:$PORT`
   - **Plan**: Free

5. **Click "Create Web Service"**

#### Step 3: Environment Variables (Optional)

Add these environment variables in Render dashboard:
- `PYTHON_VERSION`: `3.9.18`
- `MODEL_PATH`: `runs/detect/yolov8n_food101/weights/best.pt`

### Option 2: Railway (Alternative Free Platform)

**Railway** offers a simple deployment process with a free tier.

#### Step 1: Deploy on Railway

1. **Go to [railway.app](https://railway.app)** and sign up
2. **Click "New Project"** → **"Deploy from GitHub repo"**
3. **Select your repository**
4. **Railway will auto-detect Python and deploy**

#### Step 2: Configure Railway

1. **Add environment variables**:
   - `PORT`: `8000`
   - `PYTHON_VERSION`: `3.9.18`

2. **Set build command**: `pip install -r requirements.txt`
3. **Set start command**: `gunicorn food_detection_app_cloud:app --bind 0.0.0.0:$PORT`

### Option 3: Heroku (Legacy - Limited Free Tier)

**Note**: Heroku no longer offers a free tier, but here's how to deploy if you have a paid account.

#### Step 1: Prepare for Heroku

1. **Install Heroku CLI**:
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Windows
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku**:
   ```bash
   heroku login
   ```

#### Step 2: Deploy

1. **Create Heroku app**:
   ```bash
   heroku create your-food-detection-app
   ```

2. **Set buildpacks**:
   ```bash
   heroku buildpacks:set heroku/python
   ```

3. **Deploy**:
   ```bash
   git push heroku main
   ```

4. **Open the app**:
   ```bash
   heroku open
   ```

## 🔧 Model Deployment Considerations

### Large Model Files

Your YOLO model (`best.pt`) is ~6MB. For free platforms:

1. **Use Git LFS** (Large File Storage):
   ```bash
   git lfs install
   git lfs track "*.pt"
   git add .gitattributes
   git commit -m "Add LFS tracking for model files"
   ```

2. **Alternative: Host model separately**:
   - Upload to Google Drive/Dropbox
   - Download during app startup
   - Use cloud storage (AWS S3, Google Cloud Storage)

### Model Optimization

For faster deployment and inference:

1. **Convert to ONNX format**:
   ```python
   from ultralytics import YOLO
   
   model = YOLO("runs/detect/yolov8n_food101/weights/best.pt")
   model.export(format="onnx")
   ```

2. **Use smaller model variants**:
   - `yolov8n.pt` (nano) - ~6MB
   - `yolov8s.pt` (small) - ~22MB
   - `yolov8m.pt` (medium) - ~52MB

## 🌐 Custom Domain (Optional)

### Render
1. Go to your service dashboard
2. Click "Settings" → "Custom Domains"
3. Add your domain and configure DNS

### Railway
1. Go to your project dashboard
2. Click "Settings" → "Domains"
3. Add custom domain

## 📊 Monitoring and Logs

### Render
- **Logs**: Available in the dashboard
- **Metrics**: Built-in monitoring
- **Health Checks**: Automatic

### Railway
- **Logs**: Real-time in dashboard
- **Metrics**: Basic monitoring
- **Alerts**: Email notifications

## 🔍 Troubleshooting

### Common Issues

1. **Build Failures**:
   ```
   Error: No module named 'ultralytics'
   ```
   **Solution**: Ensure `requirements.txt` includes all dependencies

2. **Model Loading Errors**:
   ```
   Error: Model file not found
   ```
   **Solution**: Check model path and file size

3. **Memory Issues**:
   ```
   Error: Out of memory
   ```
   **Solution**: Use smaller model or optimize code

4. **Timeout Errors**:
   ```
   Error: Request timeout
   ```
   **Solution**: Optimize inference speed or increase timeout limits

### Performance Optimization

1. **Enable Caching**:
   ```python
   from flask_caching import Cache
   
   cache = Cache(app, config={'CACHE_TYPE': 'simple'})
   ```

2. **Image Compression**:
   ```python
   from PIL import Image
   
   def compress_image(image_path, max_size=(800, 800)):
       img = Image.open(image_path)
       img.thumbnail(max_size, Image.Resampling.LANCZOS)
       img.save(image_path, optimize=True, quality=85)
   ```

3. **Async Processing**:
   ```python
   import asyncio
   from concurrent.futures import ThreadPoolExecutor
   
   executor = ThreadPoolExecutor(max_workers=4)
   ```

## 🔐 Security Considerations

1. **File Upload Validation**:
   - Check file extensions
   - Validate file content
   - Limit file sizes

2. **Rate Limiting**:
   ```python
   from flask_limiter import Limiter
   
   limiter = Limiter(app, key_func=get_remote_address)
   
   @app.route('/upload', methods=['POST'])
   @limiter.limit("10 per minute")
   def upload_file():
       # Your code here
   ```

3. **Environment Variables**:
   - Store sensitive data in environment variables
   - Never commit API keys or secrets

## 📈 Scaling Considerations

### Free Tier Limits

- **Render**: 750 hours/month, 512MB RAM
- **Railway**: $5 credit/month, 512MB RAM
- **Heroku**: Paid plans only

### Upgrade Path

1. **Paid Plans**: More resources and features
2. **Custom VPS**: Full control and scalability
3. **Cloud Providers**: AWS, Google Cloud, Azure

## 🎉 Success Checklist

- [ ] Repository is public and accessible
- [ ] All dependencies in `requirements.txt`
- [ ] Model file is included or downloadable
- [ ] App starts without errors
- [ ] Health check endpoint works
- [ ] File upload and detection works
- [ ] Custom domain configured (optional)
- [ ] Monitoring and logging set up

## 📞 Support

- **Render**: [docs.render.com](https://docs.render.com)
- **Railway**: [docs.railway.app](https://docs.railway.app)
- **Heroku**: [devcenter.heroku.com](https://devcenter.heroku.com)

---

**Happy Deploying! 🚀🍕** 