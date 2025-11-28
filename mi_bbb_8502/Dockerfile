FROM python:3.9

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requisitos e instala las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de tu aplicación
COPY . .

# Expone el puerto 8501, que es el puerto estándar de Streamlit
EXPOSE 8501

# Comando para iniciar la aplicación de Streamlit
# --server.address=0.0.0.0 permite acceso desde fuera del contenedor
# --server.headless=true para modo sin interfaz gráfica
# --server.enableCORS=false y --server.enableXsrfProtection=false para desarrollo
CMD ["streamlit", "run", "conciliacion_bancaria.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]