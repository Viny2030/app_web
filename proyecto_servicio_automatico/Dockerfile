# ===========================================================
# BASE IMAGE ESTABLE PARA PROPHET + SKLEARN + XGBOOST
# ===========================================================
FROM python:3.10-slim-bullseye

# ===========================================================
# LOCALES UTF-8
# ===========================================================
RUN apt-get update && apt-get install -y locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ===========================================================
# DEPENDENCIAS DEL SISTEMA NECESARIAS PARA PROPHET, PMDARIMA
# ===========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    libgomp1 \
    libxml2 \
    libxml2-dev \
    libxslt1-dev \
    libssl-dev \
    libffi-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ===========================================================
# DIRECTORIO APP
# ===========================================================
WORKDIR /app

# ===========================================================
# INSTALAR DEPENDENCIAS PYTHON
# ===========================================================
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ===========================================================
# COPIAR TODA LA APLICACIÓN
# ===========================================================


COPY . /app
RUN sed -i 's/\r$//' /app/start.sh
RUN sed -i 's/\r$//' /app/selector_modelo.py
RUN chmod +x /app/start.sh


# ===========================================================
# CREAR USUARIO NO ROOT
# ===========================================================
RUN useradd -ms /bin/bash appuser
USER appuser

# ===========================================================
# EXPOSE (STREAMLIT)
# ===========================================================
EXPOSE 8501

# ===========================================================
# ENTRYPOINT
# ===========================================================
ENTRYPOINT ["/app/start.sh"]

