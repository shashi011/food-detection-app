#!/usr/bin/env python3
"""
Food Detection Application using YOLOv8
A Flask web application for detecting food items in images.
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import base64
from PIL import Image
import io
import json
from datetime import datetime
import torch

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Optimize for production
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = False

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Food classes
FOOD_CLASSES = [
    "Apple Pie", "Chocolate", "French Fries", "Hotdog", "Nachos", 
    "Pizza", "onion_rings", "pancakes", "spring_rolls", "tacos"
]

# Load YOLO model - Simple approach to bypass PyTorch 2.6+ restrictions
try:
    print("🔄 Loading YOLO model...")
    
    # Check if model file exists
    model_path = "runs/detect/yolov8n_food101/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        print(f"📁 Current working directory: {os.getcwd()}")
        print(f"📁 Available files in runs/detect/yolov8n_food101/weights/: {os.listdir('runs/detect/yolov8n_food101/weights/') if os.path.exists('runs/detect/yolov8n_food101/weights/') else 'Directory not found'}")
        model = None
    else:
        print(f"✅ Model file found at: {model_path}")
        
        # Temporarily patch torch.load to use weights_only=False
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
        
        # Load model with memory optimization
        print("🔄 Loading model with memory optimization...")
        model = YOLO(model_path)
        
        # Move model to CPU and optimize memory
        if hasattr(model, 'model'):
            model.model = model.model.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Restore original torch.load
        torch.load = original_torch_load
        
        print("✅ YOLO model loaded successfully!")
        
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    print(f"📁 Current working directory: {os.getcwd()}")
    import traceback
    traceback.print_exc()
    model = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process image with YOLO model and return results."""
    if model is None:
        print("❌ Model is None - cannot process image")
        return {"error": "Model not loaded"}
    
    try:
        print(f"🔄 Processing image: {image_path}")
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return {"error": f"Image file not found: {image_path}"}
        
        # Run inference with timeout and memory optimization
        print("🔄 Running YOLO inference...")
        
        # Ensure model is on CPU for memory efficiency
        if hasattr(model, 'model'):
            model.model = model.model.cpu()
        
        # Run inference with smaller image size to save memory
        results = model(image_path, imgsz=640)  # Limit image size
        print(f"✅ Inference completed, got {len(results)} results")
        
        # Process results
        detections = []
        for i, result in enumerate(results):
            print(f"🔄 Processing result {i+1}/{len(results)}")
            boxes = result.boxes
            if boxes is not None:
                print(f"📦 Found {len(boxes)} boxes in result {i+1}")
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = FOOD_CLASSES[class_id] if class_id < len(FOOD_CLASSES) else f"Class_{class_id}"
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': round(confidence * 100, 2),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
            else:
                print(f"📭 No boxes found in result {i+1}")
        
        # Clear memory
        del results
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"✅ Processing completed. Total detections: {len(detections)}")
        return {
            'success': True,
            'detections': detections,
            'total_detections': len(detections)
        }
        
    except Exception as e:
        print(f"❌ Error in process_image: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error processing image: {str(e)}"}

def save_annotated_image(image_path, detections, output_path):
    """Save image with bounding boxes and labels."""
    try:
        # Read image
        image = cv2.imread(image_path)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save annotated image
        cv2.imwrite(output_path, image)
        return True
        
    except Exception as e:
        print(f"Error saving annotated image: {e}")
        return False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html', food_classes=FOOD_CLASSES)

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and detection."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image
            results = process_image(filepath)
            
            if results.get('success'):
                # Save annotated image
                output_filename = f"annotated_{filename}"
                output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
                save_annotated_image(filepath, results['detections'], output_path)
                
                # Add file paths to results
                results['original_image'] = filename
                results['annotated_image'] = output_filename
                
                # Clean up temporary file
                os.remove(filepath)
                
                return jsonify(results)
            else:
                # Clean up temporary file even if processing fails
                os.remove(filepath)
                return jsonify(results)
                
        except Exception as e:
            # Clean up if file was saved before error
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing file: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/detect', methods=['POST'])
def detect_food():
    """API endpoint for food detection."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image
            results = process_image(filepath)
            
            # Clean up temporary file
            os.remove(filepath)
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    print("🍕 Food Detection Application Starting...")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Results folder: {app.config['RESULTS_FOLDER']}")
    print(f"🎯 Food classes: {len(FOOD_CLASSES)} classes")
    print("🌐 Starting Flask server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 