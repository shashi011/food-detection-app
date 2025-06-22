# 🍕 Food Detection AI Application

A comprehensive food detection application using YOLOv8 for detecting 10 different food classes in images.

## 🎯 Supported Food Classes

1. **Apple Pie** 🥧
2. **Chocolate** 🍫
3. **French Fries** 🍟
4. **Hotdog** 🌭
5. **Nachos** 🌮
6. **Pizza** 🍕
7. **Onion Rings** 🧅
8. **Pancakes** 🥞
9. **Spring Rolls** 🥢
10. **Tacos** 🌮

## 🚀 Features

- **Web Interface**: Beautiful, modern web UI with drag-and-drop functionality
- **Command Line Tool**: Fast batch processing for multiple images
- **Real-time Detection**: Instant results with confidence scores
- **Bounding Box Visualization**: Annotated images with detection boxes
- **Batch Processing**: Process entire directories of images
- **API Endpoint**: RESTful API for integration with other applications

## 📋 Requirements

- Python 3.8+
- YOLOv8 model (already trained and available in `runs/detect/yolov8n_food101/weights/best.pt`)

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model Path**:
   Ensure your YOLO model is located at:
   ```
   runs/detect/yolov8n_food101/weights/best.pt
   ```

## 🎮 Usage

### 1. Web Application

**Start the Flask server**:
```bash
python food_detection_app.py
```

**Access the application**:
- Open your browser and go to: `http://localhost:5000`
- Upload images by dragging and dropping or clicking to browse
- View real-time detection results with bounding boxes

### 2. Command Line Tool

**Single Image Detection**:
```bash
python food_detection_cli.py path/to/image.jpg -o output.jpg
```

**Batch Processing**:
```bash
python food_detection_cli.py path/to/images/folder -d output/folder
```

**List Supported Classes**:
```bash
python food_detection_cli.py --list-classes
```

**Adjust Confidence Threshold**:
```bash
python food_detection_cli.py image.jpg -c 0.7
```

### 3. API Usage

**Upload and Detect**:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/upload
```

**API Response Format**:
```json
{
  "success": true,
  "detections": [
    {
      "class_name": "Pizza",
      "confidence": 95.2,
      "bbox": [100, 150, 300, 400]
    }
  ],
  "total_detections": 1,
  "original_image": "timestamp_image.jpg",
  "annotated_image": "annotated_timestamp_image.jpg"
}
```

## 📁 File Structure

```
├── food_detection_app.py          # Flask web application
├── food_detection_cli.py          # Command-line tool
├── requirements.txt               # Python dependencies
├── templates/
│   └── index.html                # Web interface template
├── uploads/                      # Temporary uploaded files
├── results/                      # Annotated output images
└── runs/detect/yolov8n_food101/weights/
    └── best.pt                   # Trained YOLO model
```

## 🎨 Web Interface Features

- **Modern UI**: Beautiful gradient design with smooth animations
- **Drag & Drop**: Easy image upload with visual feedback
- **Real-time Results**: Instant detection with loading indicators
- **Statistics**: Total detections and unique food types
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: Clear error messages for troubleshooting

## 🔧 Configuration

### Model Settings
- **Model Path**: `runs/detect/yolov8n_food101/weights/best.pt`
- **Confidence Threshold**: Default 0.5 (50%)
- **Max File Size**: 16MB for web uploads

### Web Server Settings
- **Host**: `0.0.0.0` (accessible from any IP)
- **Port**: `5000`
- **Debug Mode**: Enabled for development

## 📊 Performance

- **Inference Speed**: ~50-100ms per image (depending on hardware)
- **Accuracy**: Trained on Food-101 dataset subset
- **Memory Usage**: ~2-4GB RAM (depending on image size)

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found**:
   ```
   ❌ Error loading YOLO model: [Errno 2] No such file or directory
   ```
   **Solution**: Ensure the model file exists at the specified path

2. **CUDA/GPU Issues**:
   ```
   ❌ CUDA out of memory
   ```
   **Solution**: Use CPU mode or reduce batch size

3. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'ultralytics'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

### Performance Tips

- **GPU Acceleration**: Install CUDA-enabled PyTorch for faster inference
- **Batch Processing**: Use CLI tool for processing multiple images
- **Image Optimization**: Resize large images before processing

## 🔄 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/upload` | POST | Upload and detect food in image |
| `/detect` | POST | API endpoint for detection |
| `/uploads/<filename>` | GET | Serve uploaded files |
| `/results/<filename>` | GET | Serve result files |

## 📈 Future Enhancements

- [ ] Real-time video detection
- [ ] Mobile app integration
- [ ] Nutritional information lookup
- [ ] Recipe suggestions based on detected foods
- [ ] Multi-language support
- [ ] Cloud deployment options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the AIML Computer Vision Food-101 Detection project.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Food-101 dataset
- Flask web framework
- OpenCV for image processing

---

**Happy Food Detection! 🍕🍔🍟** 