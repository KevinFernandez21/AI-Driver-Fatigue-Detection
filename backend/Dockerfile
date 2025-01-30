# Utilizamos la imagen ligera de Python
FROM python:3.10-slim

# Permitir que los logs se muestren en tiempo real
ENV PYTHONUNBUFFERED True

# Definir el puerto para Cloud Run
ENV PORT 8080

# Definir directorio de la app
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos estáticos
COPY static/favicon.ico static/

# Exponer el puerto 8080
EXPOSE 8080

# Ejecutar Gunicorn con UvicornWorker
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 -k uvicorn.workers.UvicornWorker main:app
