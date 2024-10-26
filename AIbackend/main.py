from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import requests
import mediapipe as mp
from io import BytesIO

app = FastAPI()

# URL de la ESP32-CAM
CAMERA_URL = "http://192.168.100.145/capture"
BUZZER_ON_URL = "http://192.168.100.145/buzzer/on"
BUZZER_OFF_URL = "http://192.168.100.145/buzzer/off"

# Inicializa MediaPipe para la detección facial
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_eye_aspect_ratio(landmarks, eye_indices):
    horizontal_dist = landmarks[eye_indices[1]].x - landmarks[eye_indices[0]].x
    vertical_dist = landmarks[eye_indices[2]].y - landmarks[eye_indices[3]].y
    return vertical_dist / horizontal_dist

@app.get("/stream")
async def stream_camera():
    # Captura imagen de la ESP32-CAM
    img_resp = requests.get(CAMERA_URL)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    
    # Convierte la imagen a formato JPEG para transmitir
    _, jpeg = cv2.imencode('.jpg', frame)
    return StreamingResponse(BytesIO(jpeg.tobytes()), media_type="image/jpeg")

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Procesa la imagen subida para detectar ojos cerrados
    image = await file.read()
    nparr = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    eyes_closed = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_ratio = calculate_eye_aspect_ratio(face_landmarks.landmark, [33, 133, 159, 145])
            right_eye_ratio = calculate_eye_aspect_ratio(face_landmarks.landmark, [362, 263, 386, 374])
            if left_eye_ratio < 0.2 and right_eye_ratio < 0.2:
                eyes_closed = True
                break

    return JSONResponse(content={"eyes_closed": eyes_closed})

@app.get("/buzzer/on")
async def buzzer_on():
    requests.get(BUZZER_ON_URL)
    return {"status": "Buzzer ON"}

@app.get("/buzzer/off")
async def buzzer_off():
    requests.get(BUZZER_OFF_URL)
    return {"status": "Buzzer OFF"}
