# Facial Emotion Recognition Web App

A web application that uses Deep Learning to recognize emotions from facial images.

## Features

- Automatic face detection in images
- Recognition of 5 emotions: Angry, Fear, Happy, Sad, Surprise
- Display confidence level for each prediction
- User-friendly interface
- Support for multiple faces in a single image

## Installation

### Step 1: Clone repository (if not already done)
```bash
git clone <repository-url>
cd Human-Face-Emotion-Classification
```

### Step 2: Install required libraries
```bash
pip install -r requirements_webapp.txt
```

Or if you want to use the original requirements.txt file:
```bash
pip install -r requirements.txt
pip install streamlit
```

### Step 3: Ensure model file exists
Check that the file `model/best_model_acc_final.pth` exists.

## Running the Application

Run the following command in terminal:

```bash
streamlit run app.py
```

The application will automatically open in your browser at: `http://localhost:8501`

## Usage Guide

1. **Upload image**: Click the "Browse files" button to select an image from your computer
2. **View results**: The application will automatically:
   - Detect all faces in the image
   - Draw colored boxes around each face
   - Display emotion and confidence level
3. **Details**: Expand each face to view detailed information

## Recognized Emotions

| Emotion | Color | Description |
|---------|-------|-------------|
| Angry | Red | Anger emotion |
| Fear | Purple | Fear emotion |
| Happy | Green | Happy emotion |
| Sad | Blue | Sad emotion |
| Surprise | Orange | Surprise emotion |

## Technology Stack

- **PyTorch**: Deep Learning framework
- **Streamlit**: Web App framework
- **OpenCV**: Image processing and face detection
- **CNN**: Custom Convolutional Neural Network

## Directory Structure

```
Human-Face-Emotion-Classification/
├── app.py                          # Main web app file
├── requirements_webapp.txt         # Web app dependencies
├── README_WEBAPP.md               # This file
├── model/
│   ├── best_model_acc_final.pth   # Trained model
│   └── training.ipynb             # Training notebook
└── data/                          # Training data (if any)
```

## Customization

### Change default port
```bash
streamlit run app.py --server.port 8080
```

### Run in production mode
```bash
streamlit run app.py --server.headless true
```

### Advanced configuration
Create a `.streamlit/config.toml` file:
```toml
[server]
port = 8501
enableCORS = false

[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## Troubleshooting

### Error: "Module not found"
```bash
pip install --upgrade -r requirements_webapp.txt
```

### Error: "Model not found"
Ensure the model file is in the correct location: `model/best_model_acc_final.pth`

### Error: "No faces detected"
- Ensure the image contains clear faces
- Faces should be frontal or slightly tilted
- Sufficient lighting for recognition

## Notes

- Model works best with clear facial images
- GPU support if available (automatic CUDA detection)
- Can process multiple faces in one image

## Contributing

All contributions are welcome! Please create a Pull Request or open an Issue.

## License

MIT License

---
