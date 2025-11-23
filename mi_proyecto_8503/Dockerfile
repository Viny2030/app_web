# Usa una imagen base oficial de Python
FROM python:3.11-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos de requerimientos e instálalos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia los archivos de la aplicación
COPY . .

# Expone el puerto interno del contenedor
EXPOSE 8503 
# Inicia Streamlit en el puerto 8503
ENTRYPOINT ["streamlit", "run", "app_auditoria.py", "--server.port=8503", "--server.enableCORS=true"]