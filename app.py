import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import sys
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from model import EmotionCNN

st.set_page_config(
    page_title="Emotion Recognition",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=5).to(device)
    
    model_path = 'model/best_model_acc_final.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class_labels = {
    0: 'Angry',
    1: 'Fear',
    2: 'Happy',
    3: 'Sad',
    4: 'Surprise'
}

emotion_colors = {
    0: (255, 0, 0),      # Red - Angry
    1: (128, 0, 128),    # Purple - Fear
    2: (0, 255, 0),      # Green - Happy
    3: (0, 0, 255),      # Blue - Sad
    4: (255, 165, 0)     # Orange - Surprise
}

def detect_and_predict_emotions(image, model, device):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, "No faces detected in the image"
    
    results = []
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        pil_img = Image.fromarray(face_roi)
        
        img_tensor = inference_transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        label_idx = pred.item()
        label = class_labels[label_idx]
        score = conf.item() * 100
        
        color = emotion_colors[label_idx]
        cv2.rectangle(img_cv2, (x, y), (x+w, y+h), color, 3)
        
        text = f"{label}: {score:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        cv2.rectangle(img_cv2, (x, y-35), (x + text_size[0] + 10, y), (0, 0, 0), -1)
        cv2.putText(img_cv2, text, (x+5, y-10), font, font_scale, color, thickness)
        
        results.append({
            'emotion': label,
            'confidence': score,
            'bbox': (x, y, w, h)
        })
    
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return img_rgb, results

class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            img_tensor = inference_transform(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            label_idx = pred.item()
            label = class_labels[label_idx]
            score = conf.item() * 100
            color = emotion_colors[label_idx]
            
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = f"{label} ({score:.0f}%)"
            text_y = y - 10 if y - 10 > 20 else y + h + 20
            cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Facial Emotion Recognition")
st.markdown("---")

model, device = load_model()

if model is None:
    st.error("Unable to load model. Please check the model file path.")
    st.stop()

with st.sidebar:
    st.header("Instructions")
    
    input_mode = st.radio(
        "Select Input Mode:",
        ["Upload Image", "Real-time Webcam"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("""
    **How to use:**
    
    **Upload Image Mode:**
    - Upload an image containing faces
    - View detection results instantly
    
    **Real-time Webcam Mode:**
    - Allow camera access when prompted
    - See live emotion detection
    - Works continuously in real-time
    
    **Recognized emotions:**
    - Angry
    - Fear
    - Happy
    - Sad
    - Surprise
    """)

if input_mode == "Upload Image":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.subheader("Detection Results")
        
        if uploaded_file is not None:
            with st.spinner("Processing..."):
                result_img, results = detect_and_predict_emotions(image, model, device)
                
                if result_img is not None:
                    st.image(result_img, caption="Detection Results", use_container_width=True)
                    
                    st.success(f"Detected {len(results)} face(s)")
                    
                    for i, res in enumerate(results, 1):
                        with st.expander(f"Face {i}"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Emotion", res['emotion'])
                            with col_b:
                                st.metric("Confidence", f"{res['confidence']:.2f}%")
                else:
                    st.warning(results)
        else:
            st.info("Please upload an image to begin")

else:
    st.subheader("Real-time Emotion Detection")
    st.info("Allow camera access when prompted. The detection will run continuously.")
    
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>Built with PyTorch and Streamlit</p>"
    "</div>",
    unsafe_allow_html=True
)
