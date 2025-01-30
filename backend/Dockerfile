# Utilizamos la imagen ligera de Python
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Permitir que las declaraciones y los mensajes de registro aparezcan inmediatamente en los registros de Knative
ENV PYTHONUNBUFFERED True

# Copiamos el código local al contenedor
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Instalamos dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalamos las dependencias necesarias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el favicon y otros archivos estáticos (si tienes alguno)
# Asegúrate de tener un archivo favicon.ico en la carpeta "static"
COPY static/favicon.ico static/

# Ejecutar Gunicorn con el adaptador UvicornWorker al iniciar el contenedor
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 -k uvicorn.workers.UvicornWorker app.main:app 