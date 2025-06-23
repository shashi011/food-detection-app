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
from ultralytics.nn.tasks import DetectionModel
from torch.serialization import add_safe_globals
import torch.nn.modules.container
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.head import Detect
import torch.nn.modules.conv
import torch.nn.modules.batchnorm
import torch.nn.modules.activation
import torch.nn.modules.pooling
import torch.nn.modules.dropout
import torch.nn.modules.normalization
import torch.nn.modules.linear
import torch.nn.modules.flatten
import torch.nn.modules.upsampling
import torch.nn.modules.padding
import ultralytics.nn.tasks
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block
import ultralytics.nn.modules.head

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Food classes
FOOD_CLASSES = [
    "Apple Pie", "Chocolate", "French Fries", "Hotdog", "Nachos", 
    "Pizza", "onion_rings", "pancakes", "spring_rolls", "tacos"
]

# 🔧 Fix for PyTorch 2.6+ weights_only issue - Comprehensive list

try:
    add_safe_globals([
        # Container modules
        torch.nn.modules.container.Sequential,
        torch.nn.modules.container.ModuleList,
        torch.nn.modules.container.ModuleDict,
        torch.nn.modules.container.ParameterList,
        torch.nn.modules.container.ParameterDict,
        
        # Convolutional layers
        torch.nn.modules.conv.Conv1d,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.conv.Conv3d,
        torch.nn.modules.conv.ConvTranspose1d,
        torch.nn.modules.conv.ConvTranspose2d,
        torch.nn.modules.conv.ConvTranspose3d,
        torch.nn.modules.conv.LazyConv1d,
        torch.nn.modules.conv.LazyConv2d,
        torch.nn.modules.conv.LazyConv3d,
        torch.nn.modules.conv.LazyConvTranspose1d,
        torch.nn.modules.conv.LazyConvTranspose2d,
        torch.nn.modules.conv.LazyConvTranspose3d,
        
        # Batch normalization
        torch.nn.modules.batchnorm.BatchNorm1d,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.batchnorm.BatchNorm3d,
        torch.nn.modules.batchnorm.LazyBatchNorm1d,
        torch.nn.modules.batchnorm.LazyBatchNorm2d,
        torch.nn.modules.batchnorm.LazyBatchNorm3d,
        
        # Activation functions
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.activation.ReLU6,
        torch.nn.modules.activation.LeakyReLU,
        torch.nn.modules.activation.PReLU,
        torch.nn.modules.activation.RReLU,
        torch.nn.modules.activation.ELU,
        torch.nn.modules.activation.CELU,
        torch.nn.modules.activation.SELU,
        torch.nn.modules.activation.GLU,
        torch.nn.modules.activation.GELU,
        torch.nn.modules.activation.SiLU,
        torch.nn.modules.activation.Hardswish,
        torch.nn.modules.activation.Mish,
        torch.nn.modules.activation.Sigmoid,
        torch.nn.modules.activation.Tanh,
        torch.nn.modules.activation.Softmax,
        torch.nn.modules.activation.LogSoftmax,
        
        # Pooling layers
        torch.nn.modules.pooling.MaxPool1d,
        torch.nn.modules.pooling.MaxPool2d,
        torch.nn.modules.pooling.MaxPool3d,
        torch.nn.modules.pooling.AvgPool1d,
        torch.nn.modules.pooling.AvgPool2d,
        torch.nn.modules.pooling.AvgPool3d,
        torch.nn.modules.pooling.AdaptiveMaxPool1d,
        torch.nn.modules.pooling.AdaptiveMaxPool2d,
        torch.nn.modules.pooling.AdaptiveMaxPool3d,
        torch.nn.modules.pooling.AdaptiveAvgPool1d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool3d,
        torch.nn.modules.pooling.FractionalMaxPool2d,
        torch.nn.modules.pooling.FractionalMaxPool3d,
        torch.nn.modules.pooling.LPPool1d,
        torch.nn.modules.pooling.LPPool2d,
        
        # Dropout layers
        torch.nn.modules.dropout.Dropout,
        torch.nn.modules.dropout.Dropout1d,
        torch.nn.modules.dropout.Dropout2d,
        torch.nn.modules.dropout.Dropout3d,
        torch.nn.modules.dropout.AlphaDropout,
        
        # Normalization layers
        torch.nn.modules.normalization.LocalResponseNorm,
        torch.nn.modules.normalization.CrossMapLRN2d,
        torch.nn.modules.normalization.GroupNorm,
        torch.nn.modules.normalization.LayerNorm,
        torch.nn.modules.normalization.LazyInstanceNorm1d,
        torch.nn.modules.normalization.LazyInstanceNorm2d,
        torch.nn.modules.normalization.LazyInstanceNorm3d,
        
        # Linear layers
        torch.nn.modules.linear.Linear,
        torch.nn.modules.linear.Bilinear,
        torch.nn.modules.linear.LazyLinear,
        
        # Flatten and reshape
        torch.nn.modules.flatten.Flatten,
        torch.nn.modules.flatten.Unflatten,
        
        # Upsampling
        torch.nn.modules.upsampling.Upsample,
        torch.nn.modules.upsampling.UpsamplingNearest2d,
        torch.nn.modules.upsampling.UpsamplingBilinear2d,
        
        # Padding
        torch.nn.modules.padding.ReflectionPad1d,
        torch.nn.modules.padding.ReflectionPad2d,
        torch.nn.modules.padding.ReflectionPad3d,
        torch.nn.modules.padding.ReplicationPad1d,
        torch.nn.modules.padding.ReplicationPad2d,
        torch.nn.modules.padding.ReplicationPad3d,
        torch.nn.modules.padding.ZeroPad2d,
        torch.nn.modules.padding.ConstantPad1d,
        torch.nn.modules.padding.ConstantPad2d,
        torch.nn.modules.padding.ConstantPad3d,
        
        # Ultralytics modules - Comprehensive list
        ultralytics.nn.tasks.DetectionModel,
        ultralytics.nn.modules.conv.Conv,
        ultralytics.nn.modules.block.C2f,
        ultralytics.nn.modules.block.SPPF,
        ultralytics.nn.modules.head.Detect,
        DetectionModel,
        Conv,
        C2f,
        SPPF,
        Detect
    ])
except Exception as e:
    print(f"⚠️ Failed to register safe globals: {e}")

# Load YOLO model
try:
    model = YOLO("runs/detect/yolov8n_food101/weights/best.pt")
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process image with YOLO model and return results."""
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
                    
                    # Get class name
                    class_name = FOOD_CLASSES[class_id] if class_id < len(FOOD_CLASSES) else f"Class_{class_id}"
                    
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