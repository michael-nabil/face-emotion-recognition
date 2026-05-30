from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import base64

app = FastAPI()

# Allow CORS so frontend can connect if served from a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_face(face_roi: np.ndarray, target_size: tuple = (244, 244)):
    # 1. Converting BGR (OpenCV default) to Grayscale
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Duplicating the grayscale channel to mimic RGB (H, W, 3)
    gray_3ch = cv2.merge([gray, gray, gray])
    
    # 3. Resizing to the model's required input size
    resized = cv2.resize(gray_3ch, target_size, interpolation=cv2.INTER_AREA)
    
    # 4. Converting to float array and normalize
    roi_array = img_to_array(resized) / 255.0
    
    # 5. Adding batch dimension -> (1, 244, 244, 3)
    return np.expand_dims(roi_array, axis=0)

face_detector = cv2.CascadeClassifier(r"..\..\models\haarcascade_frontalface_default.xml")
emotion_classifier = load_model(r'..\..\models\model_phase2.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.websocket("/ws/predict_emotion")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 1. Receive Base64 image string from frontend
            data = await websocket.receive_text()
            
            # 2. Decode the Base64 string to an OpenCV image
            # The data usually looks like "data:image/jpeg;base64,/9j/4AAQ..."
            encoded_data = data.split(',')[1] if ',' in data else data
            img_bytes = base64.b64decode(encoded_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 3. converting the frame to gray scale (required by the haarcascade detector)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray)
            
            results = []
            # roi: Region Of Interest (a.k.a Anchor)
            for (x, y, w, h) in faces:
                raw_face_roi = frame[y:y+h, x:x+w]

                if raw_face_roi.size != 0:
                    processed_roi = preprocess_face(raw_face_roi, target_size=(244, 244))

                    # verbose=0 speeds up inference by removing progress bars in the terminal
                    prediction = emotion_classifier.predict(processed_roi, verbose=0)[0]
                    label = emotion_labels[prediction.argmax()]
                    
                    results.append({
                        "box": [int(x), int(y), int(w), int(h)],
                        "emotion": label
                    })

            # 4. Send JSON results back to the frontend
            await websocket.send_json({"results": results})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")