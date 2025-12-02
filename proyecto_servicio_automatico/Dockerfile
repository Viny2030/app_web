# Dockerfile
# Usar una imagen base de Python 3.10 ligera
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# 1. Copiar las dependencias e instalarlas
COPY requirements.txt .
# Instalar dependencias, incluyendo build-essential para librerías que necesitan compilación (como numpy/pandas)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2. Copiar los archivos del Dashboard (Se asume que test_run.py los coloca aquí)
# Estos son los archivos generados por generar_dashboard.py
COPY dashboard.py .
COPY dataset.csv .

# 3. Streamlit corre en el puerto 8501 por defecto
EXPOSE 8501

# 4. Comando para iniciar la aplicación Streamlit
# --server.enableCORS=false es útil para despliegues en contenedores
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.enableCORS=false"]