# Human Face Emotion Classification

A Deep Learning project for recognizing emotions from facial images.

## Introduction

A web application that uses a trained CNN (Convolutional Neural Network) model to classify 5 emotions: Angry, Fear, Happy, Sad, and Surprise.

## Dataset

Dataset link: https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions

## Features

- Automatic face detection in images
- Recognition of 5 emotions: Angry, Fear, Happy, Sad, Surprise
- Display confidence level for each prediction
- User-friendly interface
- Support for multiple faces in a single image

## Installation

### Step 1: Download source code
1. Download the project ZIP file
2. Extract to your desired folder:
```bash
cd Human-Face-Emotion-Classification
```

### Step 2: Install required libraries
```bash
pip install -r requirements.txt
```

### Step 3: Verify model file
Ensure the file `model/best_model_acc_final.pth` exists in the project directory.

## Running the Web Application

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
├── evaluate.py                     # Model evaluation file
├── requirements_webapp.txt         # Web app dependencies
├── requirements.txt                # Project dependencies
├── README_WEBAPP.md               # Web app documentation
├── README.md                      # This file
├── model/
│   ├── best_model_acc_final.pth   # Trained model
│   ├── best_model_loss.pth        # Model checkpoint
│   ├── model.py                   # Model architecture definition
│   ├── inference.py               # Inference code
│   └── training.ipynb             # Training notebook
└── test/                          # Test data
    ├── angry/
    ├── fear/
    ├── happy/
    ├── sad/
    └── surprise/
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

**Developed with PyTorch and Streamlit**
