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

        # Aquí puedes imprimir el tamaño de la imagen para verificar
        print(f"Imagen recibida con tamaño: {image.shape}")

        # Realizar la predicción con el modelo YOLO
        results = model(image)

        # Procesar los resultados de la predicción
        predictions = []
        
        # Si el modelo no encuentra objetos, devuelve una lista vacía
        if results and results[0].boxes:
            # Recorrer los resultados de las predicciones
            for box in results[0].boxes:
                prediction = {
                    "class": box.cls.tolist(),
                    "confidence": box.conf.tolist(),
                    "coordinates": box.xywh.tolist()
                }
                predictions.append(prediction)
        # Verificar si se ha detectado ojos cerrados o abiertos
        status = "No se detectaron ojos"
        print(predictions)
        # Estado por defecto
        eyes_open = 0
        if predictions:
            for pred in predictions:
                # Verificar que la clase es la de los ojos (1.0)
                if pred['class'][0] == 1:  
                    if pred['confidence'][0] > 0.4:
                        eyes_open += 1  # Ojo abierto
                    else:
                        continue  # Ojo cerrado, pero no incrementamos el contador

            # Verificar si ambos ojos están abiertos
            if eyes_open == 2:
                status = "Ambos ojos abiertos"
            elif eyes_open == 1:
                status = "Un ojo abierto"
            else:
                status = "Ojos cerrados"

        # Retornar el estado de los ojos con una predicción
        return JSONResponse(content={"status": status})

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return JSONResponse(status_code=500, content={"message": f"Error en el servidor: {str(e)}"})
