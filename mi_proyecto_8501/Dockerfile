# 1. Base Image (usar una estándar como 3.9)
FROM python:3.9-slim

# 2. Crea el directorio de trabajo
WORKDIR /app

# 3. Copia e instala SOLO los requisitos primero (para optimizar cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia el resto de tu código (app.py, conciliacion_bancaria.py, .csv, etc.)
COPY . .

# 5. Puertos
EXPOSE 8501

# No se usa CMD ni ENTRYPOINT, se usa el 'command' en docker-compose