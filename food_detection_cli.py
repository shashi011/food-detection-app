#!/usr/bin/env python3
"""
Command-line Food Detection Tool using YOLOv8
"""

import argparse
import cv2
import os
from ultralytics import YOLO
import json
from datetime import datetime
import sys

# Food classes
FOOD_CLASSES = [
    "Apple Pie", "Chocolate", "French Fries", "Hotdog", "Nachos", 
    "Pizza", "onion_rings", "pancakes", "spring_rolls", "tacos"
]

def load_model():
    """Load the YOLO model."""
    try:
        model = YOLO("runs/detect/yolov8n_food101/weights/best.pt")
        print("✅ YOLO model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return None

def detect_food(model, image_path, output_path=None, confidence_threshold=0.5):
    """Detect food items in an image."""
    if model is None:
        print("❌ Model not loaded!")
        return None
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        print(f"🔍 Analyzing image: {image_path}")
        
        # Run inference
        results = model(image_path, conf=confidence_threshold)
        
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
        
        print(f"✅ Found {len(detections)} food items!")
        
        # Display results
        display_results(detections)
        
        # Save annotated image if output path is provided
        if output_path:
            save_annotated_image(image_path, detections, output_path)
        
        return detections
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

def display_results(detections):
    """Display detection results in a formatted way."""
    if not detections:
        print("❌ No food items detected!")
        return
    
    print("\n" + "="*50)
    print("🍕 FOOD DETECTION RESULTS")
    print("="*50)
    
    # Group by food type
    food_counts = {}
    for detection in detections:
        food_name = detection['class_name']
        if food_name not in food_counts:
            food_counts[food_name] = []
        food_counts[food_name].append(detection['confidence'])
    
    # Display summary
    print(f"📊 Total detections: {len(detections)}")
    print(f"🍽️  Unique food types: {len(food_counts)}")
    print()
    
    # Display each food type
    for food_name, confidences in food_counts.items():
        avg_confidence = sum(confidences) / len(confidences)
        print(f"🍕 {food_name}:")
        print(f"   Count: {len(confidences)}")
        print(f"   Avg Confidence: {avg_confidence:.1f}%")
        print(f"   Max Confidence: {max(confidences):.1f}%")
        print()

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
        print(f"✅ Annotated image saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving annotated image: {e}")

def batch_detect(model, input_dir, output_dir=None, confidence_threshold=0.5):
    """Detect food in all images in a directory."""
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Find all image files
    image_files = []
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(input_dir, file))
    
    if not image_files:
        print(f"❌ No image files found in: {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    print("="*50)
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        # Generate output path if specified
        output_path = None
        if output_dir:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"annotated_{base_name}.jpg")
        
        # Detect food
        detections = detect_food(model, image_path, output_path, confidence_threshold)
        
        if detections:
            print(f"✅ Detected {len(detections)} food items")
        else:
            print("❌ No detections")

def main():
    parser = argparse.ArgumentParser(description='Food Detection using YOLOv8')
    parser.add_argument('--list-classes', action='store_true', 
                       help='List supported food classes')
    
    # Only require input if not listing classes
    if '--list-classes' in sys.argv:
        args = parser.parse_args()
    else:
        parser.add_argument('input', help='Input image file or directory')
        parser.add_argument('-o', '--output', help='Output image path (for single image)')
        parser.add_argument('-d', '--output-dir', help='Output directory (for batch processing)')
        parser.add_argument('-c', '--confidence', type=float, default=0.5, 
                           help='Confidence threshold (default: 0.5)')
        args = parser.parse_args()
    
    # List supported classes
    if args.list_classes:
        print("🍽️ Supported Food Classes:")
        for i, food_class in enumerate(FOOD_CLASSES, 1):
            print(f"  {i}. {food_class}")
        return
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single image processing
        detections = detect_food(model, args.input, args.output, args.confidence)
        
        # Save results to JSON if output is specified
        if args.output and detections:
            json_path = args.output.replace('.jpg', '.json').replace('.png', '.json')
            with open(json_path, 'w') as f:
                json.dump(detections, f, indent=2)
            print(f"✅ Results saved to: {json_path}")
            
    elif os.path.isdir(args.input):
        # Batch processing
        batch_detect(model, args.input, args.output_dir, args.confidence)
    else:
        print(f"❌ Input path not found: {args.input}")

if __name__ == "__main__":
    main() 