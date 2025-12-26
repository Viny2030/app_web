FROM python:3.9-slim
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar primero solo los archivos necesarios para caché
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar dependencias adicionales para generación de dashboards
RUN pip install --no-cache-dir \
    openpyxl \
    pandas \
    plotly \
    scikit-learn \
    xlrd \
    openpyxl

# Crear directorios necesarios
RUN mkdir -p /app/data/uploads /app/models /app/reports /app/temp_upload /app/dashboards

# Copiar el resto del código
COPY . .

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONPATH="${PYTHONPATH}:/app/src"

# Puerto expuesto
EXPOSE 8501

# Comando para ejecutar la aplicación principal
CMD ["streamlit", "run", "dashboard_generator.py", "--server.port=8501", "--server.address=0.0.0.0"]
