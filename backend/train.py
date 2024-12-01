from ultralytics import YOLO

# Cargar el modelo preentrenado de YOLOv5
model = YOLO("yolov8l.pt")  # También puedes usar otros modelos, como "yolov5m.pt", etc.

# Entrenar el modelo
model.train(data="yolo.yaml", epochs=10, batch=16, imgsz=512)