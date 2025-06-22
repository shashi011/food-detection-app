#!/bin/bash

# 🚀 Food Detection App Deployment Script
# This script helps you prepare and deploy your food detection app

echo "🍕 Food Detection App Deployment Script"
echo "========================================"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: Food Detection App"
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Check if model file exists
if [ ! -f "runs/detect/yolov8n_food101/weights/best.pt" ]; then
    echo "⚠️  Warning: YOLO model file not found at expected path"
    echo "   Expected: runs/detect/yolov8n_food101/weights/best.pt"
    echo "   The app will use a fallback model for deployment"
else
    echo "✅ YOLO model file found"
fi

# Check required files
echo "📋 Checking required files..."

required_files=(
    "food_detection_app_cloud.py"
    "requirements.txt"
    "render.yaml"
    "templates/index.html"
)

missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    else
        echo "✅ $file"
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

echo ""
echo "🎯 Deployment Options:"
echo "1. Render (Recommended - Free)"
echo "2. Railway (Alternative - Free)"
echo "3. Heroku (Paid)"
echo "4. Manual deployment"
echo ""

read -p "Choose deployment option (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Deploying to Render..."
        echo ""
        echo "📋 Steps to deploy on Render:"
        echo "1. Go to https://render.com and sign up"
        echo "2. Click 'New +' → 'Web Service'"
        echo "3. Connect your GitHub repository"
        echo "4. Configure:"
        echo "   - Name: food-detection-ai"
        echo "   - Environment: Python 3"
        echo "   - Build Command: pip install -r requirements.txt"
        echo "   - Start Command: gunicorn food_detection_app_cloud:app --bind 0.0.0.0:\$PORT"
        echo "   - Plan: Free"
        echo "5. Click 'Create Web Service'"
        echo ""
        echo "🔗 Your app will be available at: https://your-app-name.onrender.com"
        ;;
    2)
        echo ""
        echo "🚀 Deploying to Railway..."
        echo ""
        echo "📋 Steps to deploy on Railway:"
        echo "1. Go to https://railway.app and sign up"
        echo "2. Click 'New Project' → 'Deploy from GitHub repo'"
        echo "3. Select your repository"
        echo "4. Railway will auto-detect Python and deploy"
        echo "5. Add environment variables:"
        echo "   - PORT: 8000"
        echo "   - PYTHON_VERSION: 3.9.18"
        echo ""
        echo "🔗 Your app will be available at: https://your-app-name.railway.app"
        ;;
    3)
        echo ""
        echo "🚀 Deploying to Heroku..."
        echo ""
        echo "📋 Steps to deploy on Heroku:"
        echo "1. Install Heroku CLI:"
        echo "   brew tap heroku/brew && brew install heroku"
        echo "2. Login: heroku login"
        echo "3. Create app: heroku create your-food-detection-app"
        echo "4. Set buildpack: heroku buildpacks:set heroku/python"
        echo "5. Deploy: git push heroku main"
        echo "6. Open: heroku open"
        echo ""
        echo "🔗 Your app will be available at: https://your-app-name.herokuapp.com"
        ;;
    4)
        echo ""
        echo "📋 Manual Deployment Steps:"
        echo "1. Push your code to GitHub:"
        echo "   git remote add origin https://github.com/YOUR_USERNAME/food-detection-app.git"
        echo "   git push -u origin main"
        echo "2. Choose your preferred platform"
        echo "3. Follow the platform-specific instructions"
        echo "4. Configure environment variables"
        echo "5. Deploy and test"
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "📚 For detailed instructions, see DEPLOYMENT_GUIDE.md"
echo "🔧 For troubleshooting, check the deployment guide"
echo ""
echo "🎉 Happy Deploying! 🚀🍕" 