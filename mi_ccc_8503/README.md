# Auditoría de Productos en Proceso (WIP)

Aplicación web Streamlit para auditar y analizar productos en proceso de producción.

## 🚀 Características

- **Carga de datos desde Backblaze B2**: Integración con almacenamiento en la nube (con fallback a datos simulados)
- **Auditoría interactiva**: Parámetros configurables en tiempo real
- **Visualizaciones**: Gráficos interactivos con matplotlib y seaborn
- **Exportación de reportes**: Descarga de resultados en formato CSV

## 📋 Requisitos

- Python 3.11 o superior
- Dependencias listadas en `requirements.txt`

## 🔧 Instalación

1. Clonar o descargar el proyecto
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar credenciales en `.streamlit/secrets.toml`:
```toml
[b2]
key_id = "tu_key_id"
application_key = "tu_application_key"
bucket_name = "tu_bucket"
endpoint_url_raw = "s3.us-east-005.backblazeb2.com"
```

## ▶️ Ejecución

### Modo local:
```bash
streamlit run app_auditoria.py
```

### Con Docker:

**Opción 1: Docker Compose (recomendado)**
```bash
docker-compose up -d
```

**Opción 2: Docker directamente**
```bash
# Construir la imagen
docker build -t app-auditoria .

# Ejecutar el contenedor
docker run -d -p 8503:8503 --name auditoria-wip app-auditoria
```

**Ver logs del contenedor:**
```bash
docker logs -f auditoria-wip
```

**Detener el contenedor:**
```bash
# Con docker-compose
docker-compose down

# Con docker directamente
docker stop auditoria-wip
docker rm auditoria-wip
```

La aplicación estará disponible en:
- `http://localhost:8501` (ejecución local)
- `http://localhost:8503` (Docker)

## 📁 Estructura del Proyecto

```
mi_proyecto_8503/
├── app_auditoria.py          # Aplicación principal Streamlit
├── datos_auditoria.py        # Módulo de lógica de datos
├── requirements.txt          # Dependencias
├── Dockerfile                # Configuración Docker
├── .streamlit/
│   ├── config.toml           # Configuración Streamlit
│   └── secrets.toml          # Credenciales (no versionar)
└── README.md                 # Este archivo
```

## 🎛️ Parámetros de Auditoría

La aplicación permite configurar:

1. **Umbral Mínimo de Avance (%)**: Para detectar avances lentos en Producto A
2. **Cantidad Mínima para Ensamblaje**: Para validar cantidades en etapa de ensamblaje

## 📊 Funcionalidades

- Detección de anomalías en procesos de producción
- Alertas heurísticas configurables
- Visualizaciones de avance por etapa
- Análisis de cantidad vs avance
- Exportación de reportes de anomalías

## 🔒 Seguridad

- Las credenciales se almacenan en `.streamlit/secrets.toml` (no versionar)
- El archivo `secrets.toml` está incluido en `.gitignore`

## 📝 Notas

- Si no se configuran las credenciales de B2, la aplicación usará datos simulados automáticamente
- Los datos simulados se generan con Faker (localización: es_AR)

