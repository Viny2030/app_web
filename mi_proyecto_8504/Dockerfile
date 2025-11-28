# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar e instalar las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos de la aplicación
COPY . .

# Comando para ejecutar la aplicación de Streamlit
EXPOSE 8080
CMD ["streamlit", "run", "auditoria_prestamos.py", "--server.port=8080", "--server.address=0.0.0.0"]