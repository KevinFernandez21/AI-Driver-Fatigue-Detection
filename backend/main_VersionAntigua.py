from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io

# Crear la instancia de la aplicación FastAPI
app = FastAPI()

# Configurar middleware CORS para permitir las peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las direcciones de origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todas las cabeceras
)

# Cargar el modelo YOLO previamente entrenado
model = YOLO("./detect/train4/weights/best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Log de depuración
        print(f"Recibiendo archivo: {file.filename}")

        # Leer la imagen recibida
        image_data = await file.read()
        print("Imagen recibida, procesando...")

        # Convertir los datos de la imagen en un formato compatible con OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print("Error: la imagen no se pudo decodificar")
            return JSONResponse(status_code=400, content={"message": "Error en la imagen recibida."})

        # Redimensionar la imagen al tamaño esperado por el modelo
        resized_image = cv2.resize(image, (512, 512))

        # Realizar la predicción con el modelo YOLO
        results = model(resized_image)

        # Procesar los resultados de la predicción
        predictions = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                predictions.append({
                    "class": box.cls.tolist(),
                    "confidence": box.conf.tolist(),
                    "coordinates": box.xywh.tolist()
                })

        # Análisis del estado de los ojos
        eyes_open = 0
        if predictions:
            for pred in predictions:
                # Verificar que la clase detectada es la de los ojos abiertos (1.0)
                if pred["class"][0] == 1 and pred["confidence"][0] > 0.8:
                    eyes_open += 1

        # Determinar el estado de los ojos
        if eyes_open == 2:
            status = "Ambos ojos abiertos"
        elif eyes_open == 1:
            status = "Un ojo abierto"
        else:
            status = "Ojos cerrados"

        # Devolver la respuesta al cliente
        response = {
            "status": status,
            "predictions": predictions,
            "details": {
                "eyes_open_count": eyes_open,
                "total_detections": len(predictions)
            }
        }
        return JSONResponse(content=response)

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return JSONResponse(status_code=500, content={"message": f"Error en el servidor: {str(e)}"})
