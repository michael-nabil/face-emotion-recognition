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
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # Note: verbose=0 speeds up inference by removing progress bars in the terminal
                    prediction = emotion_classifier.predict(roi, verbose=0)[0]
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