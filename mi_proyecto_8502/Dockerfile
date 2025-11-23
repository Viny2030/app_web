FROM python:3.9

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requisitos e instala las dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copia el resto de los archivos de tu aplicación
COPY . .

# Expone el puerto 8080, que es el que Cloud Run usará
EXPOSE 8080

# Comando para iniciar la aplicación de Streamlit
CMD ["streamlit", "run", "conciliacion_bancaria.py", "--server.port=8080", "--server.enableCORS=false"]