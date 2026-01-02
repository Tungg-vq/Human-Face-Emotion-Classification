import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os

from model import EmotionCNN 
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")


MODEL_PATH = os.path.join(script_dir, 'best_model_acc_final.pth')
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' 


class_labels = {
    0: 'Angry', 
    1: 'Fear', 
    2: 'Happy', 
    3: 'Sad', 
    4: 'Surprise', 
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" {device}")


model = EmotionCNN(num_classes=5).to(device)


try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() 
    print("model loaded")
except Exception as e:
    print(f"model not found: {e}")
    exit()


inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


cap = cv2.VideoCapture(0) 
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

print("camera turning on..., q to quit")

while True:
    ret, frame = cap.read()
    if not ret: break

    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    
    faces = face_cascade.detectMultiScale(
        gray_frame, 
        scaleFactor=1.05, 
        minNeighbors=3,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        
        face_roi = frame[y:y+h, x:x+w]
        
        
        pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        
        
        img_tensor = inference_transform(pil_img).unsqueeze(0).to(device)
        
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
        
        idx = pred.item()
        score = conf.item() * 100
        label_text = f"{class_labels[idx]} ({score:.0f}%)"

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        
        text_y = y - 10 if y - 10 > 20 else y + h + 20
        cv2.putText(frame, label_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
  
    cv2.imshow('Emotion Detection - q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()