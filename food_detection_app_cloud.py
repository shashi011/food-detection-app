#!/usr/bin/env python3
"""
Food Detection Application using YOLOv8 - Cloud Optimized Version
A Flask web application for detecting food items in images, optimized for cloud deployment.
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
import tempfile

app = Flask(__name__)

# Configuration for cloud deployment
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Use temporary directories for cloud deployment
TEMP_DIR = tempfile.gettempdir()
UPLOAD_FOLDER = os.path.join(TEMP_DIR, 'food_detection_uploads')
RESULTS_FOLDER = os.path.join(TEMP_DIR, 'food_detection_results')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Food classes
FOOD_CLASSES = [
    "Apple Pie", "Chocolate", "French Fries", "Hotdog", "Nachos", 
    "Pizza", "onion_rings", "pancakes", "spring_rolls", "tacos"
]

# Global model variable
model = None

def load_model():
    """Load the YOLO model with error handling for cloud deployment."""
    global model
    try:
        # Try to load from the expected path
        model_path = "runs/detect/yolov8n_food101/weights/best.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print("✅ YOLO model loaded successfully!")
        else:
            print("⚠️  Model not found at expected path. Using default YOLO model.")
            # Fallback to a smaller model for cloud deployment
            model = YOLO("yolov8n.pt")
        return True
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        print("⚠️  Using fallback detection method")
        return False

def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process image with YOLO model and return results."""
    global model
    
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Run inference
        results = model(image_path)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name (handle both custom and COCO classes)
                    if class_id < len(FOOD_CLASSES):
                        class_name = FOOD_CLASSES[class_id]
                    else:
                        # Map COCO classes to food-related names
                        coco_food_mapping = {
                            0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle",
                            # Add more mappings as needed
                        }
                        class_name = coco_food_mapping.get(class_id, f"Object_{class_id}")
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': round(confidence * 100, 2),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        return {
            'success': True,
            'detections': detections,
            'total_detections': len(detections)
        }
        
    except Exception as e:
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
                
                # Clean up uploaded file after processing
                try:
                    os.remove(filepath)
                except:
                    pass
                
                return jsonify(results)
            else:
                return jsonify(results)
                
        except Exception as e:
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

@app.route('/health')
def health_check():
    """Health check endpoint for deployment platforms."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'food_classes': len(FOOD_CLASSES)
    })

@app.route('/api/detect', methods=['POST'])
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
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

# Initialize model on startup
@app.before_first_request
def initialize_model():
    """Initialize the model before the first request."""
    load_model()

if __name__ == '__main__':
    print("🍕 Food Detection Application Starting (Cloud Version)...")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Results folder: {app.config['RESULTS_FOLDER']}")
    print(f"🎯 Food classes: {len(FOOD_CLASSES)} classes")
    
    # Load model
    load_model()
    
    print("🌐 Starting Flask server...")
    
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 