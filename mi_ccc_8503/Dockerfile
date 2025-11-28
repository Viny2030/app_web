# Usa una imagen base oficial de Python
FROM python:3.11-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos de requerimientos e instálalos
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia los archivos de la aplicación
COPY app_auditoria.py .
COPY datos_auditoria.py .
COPY .streamlit/ .streamlit/

# Expone el puerto interno del contenedor
EXPOSE 8503

# Configura variables de entorno para Streamlit
ENV STREAMLIT_SERVER_PORT=8503
ENV STREAMLIT_SERVER_ENABLE_CORS=true
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Inicia Streamlit
ENTRYPOINT ["streamlit", "run", "app_auditoria.py", "--server.port=8503", "--server.enableCORS=true", "--server.headless=true"]