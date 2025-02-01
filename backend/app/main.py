from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import datetime
import os
from ultralytics import YOLO

app = FastAPI()

# Configurar CORS para permitir solicitudes desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Lista para almacenar logs
logs = []

# Cargar el modelo YOLO
try:
    model = YOLO("./detect/train4/weights/best.pt")  
    print("✅ Modelo YOLO cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo YOLO: {e}")
    model = None

# Ruta de almacenamiento de imágenes
IMAGE_FOLDER = "static"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)


@app.post("/predict")
async def predict(file: UploadFile = File(...), esp32_id: str = Form(...), esp32_ip: str = Form(...)):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Procesar la imagen
    image_data = await file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"message": "Error al decodificar imagen."})

    # Realizar la predicción con YOLO
    results = model(image) if model else None
    predictions = []
    
    if results and results[0].boxes:
        for box in results[0].boxes:
            predictions.append({
                "class": box.cls.tolist(),
                "confidence": box.conf.tolist(),
                "coordinates": box.xywh.tolist()
            })

    # Si no hay detecciones, indicar que no hubo detección en lugar de asumir "Ojos cerrados"
    if not predictions:
        status = "No se detectaron rostros u ojos"
        eyes_open = 0
    else:
        eyes_open = sum(1 for pred in predictions if pred["class"][0] == 1 and pred["confidence"][0] > 0.4)

        if eyes_open == 2:
            status = "Ambos ojos abiertos"
        elif eyes_open == 1:
            status = "Un ojo abierto"
        else:
            status = "Ojos cerrados"

    # Guardar el log con la IP del ESP32
    log_entry = {
        "esp32_id": esp32_id,
        "esp32_ip": esp32_ip,
        "status": status,
        "timestamp": timestamp
    }
    logs.append(log_entry)

    return JSONResponse(content=log_entry)



@app.get("/logs")
async def get_logs():
    """Devuelve los últimos 10 registros."""
    return JSONResponse(content={"logs": logs[-10:]})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)