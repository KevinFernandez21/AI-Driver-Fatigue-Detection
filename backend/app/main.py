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
async def predict(file: UploadFile = File(...), esp32_id: str = Form(...)):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Leer la imagen recibida
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"message": "Error al decodificar imagen."})

        # Guardar la imagen original
        image_path = f"{IMAGE_FOLDER}/{esp32_id}_{timestamp.replace(':', '-')}.jpg"
        cv2.imwrite(image_path, image)

        # Redimensionar la imagen para la predicción
        resized_image = cv2.resize(image, (512, 512))

        # Realizar la predicción con YOLO
        results = model(resized_image) if model else None

        # Procesar los resultados
        predictions = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                predictions.append({
                    "class": box.cls.tolist(),
                    "confidence": box.conf.tolist(),
                    "coordinates": box.xywh.tolist()
                })

        # Determinar el estado de los ojos
        eyes_open = sum(1 for pred in predictions if pred["class"][0] == 1 and pred["confidence"][0] > 0.4)
        status = "Ambos ojos abiertos" if eyes_open == 2 else "Un ojo abierto" if eyes_open == 1 else "Ojos cerrados"

        # Crear el log de la detección
        log_entry = {
            "esp32_id": esp32_id,
            "status": status,
            "timestamp": timestamp,
            "image_url": image_path,
            "predictions": predictions
        }
        logs.append(log_entry)

        return JSONResponse(content=log_entry)

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error interno: {str(e)}"})


@app.get("/logs")
async def get_logs():
    """Devuelve los últimos 10 registros."""
    return JSONResponse(content={"logs": logs[-10:]})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)