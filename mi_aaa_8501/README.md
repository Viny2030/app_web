# Dashboard de Auditoría de Inversiones

Este es un dashboard interactivo para el análisis de inversiones utilizando Streamlit y técnicas de machine learning.

## Despliegue en Render

Sigue estos pasos para desplegar la aplicación en Render:

1. **Preparar el repositorio**
   - Asegúrate de que tu código esté en un repositorio de GitHub, GitLab o Bitbucket.

2. **Crear una cuenta en Render**
   - Ve a [Render](https://render.com/) y crea una cuenta si aún no tienes una.

3. **Crear un nuevo servicio Web en Render**
   - Haz clic en "New" y selecciona "Web Service".
   - Conecta tu repositorio donde está alojado este proyecto.

4. **Configurar el servicio**
   - **Name**: `auditoria-inversiones` (o el nombre que prefieras)
   - **Region**: Selecciona la región más cercana a ti
   - **Branch**: `main` o la rama que desees desplegar
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

5. **Configurar variables de entorno**
   - Ve a la pestaña "Environment" y añade las siguientes variables:
     - `B2_KEY_ID`: Tu ID de clave de Backblaze B2
     - `B2_APPLICATION_KEY`: Tu clave de aplicación de Backblaze B2
     - `B2_BUCKET_NAME`: Nombre de tu bucket en Backblaze B2
     - `B2_ENDPOINT_URL_RAW`: URL de tu endpoint de Backblaze B2 (sin el 'https://' inicial)

6. **Desplegar**
   - Haz clic en "Create Web Service"
   - Render comenzará a construir y desplegar tu aplicación

## Desarrollo Local

1. Clona el repositorio:
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd mi_proyecto_8501
   ```

2. Crea un entorno virtual y actívalo:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: .\venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Crea un archivo `secrets.toml` en la carpeta `.streamlit/` con tus credenciales:
   ```toml
   [b2]
   key_id = "tu_key_id"
   application_key = "tu_application_key"
   bucket_name = "tu_bucket_name"
   endpoint_url_raw = "tu_endpoint_url"
   ```

5. Ejecuta la aplicación localmente:
   ```bash
   streamlit run app.py
   ```

## Estructura del Proyecto

- `app.py`: Aplicación principal de Streamlit
- `requirements.txt`: Dependencias del proyecto
- `render.yaml`: Configuración para el despliegue en Render
- `setup.sh`: Script de configuración
- `runtime.txt`: Especifica la versión de Python

## Licencia

Este proyecto está bajo la licencia MIT.
