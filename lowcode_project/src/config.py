import os
from dotenv import load_dotenv
from pathlib import Path

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de Backblaze B2
B2_KEY_ID = os.getenv('B2_KEY_ID')
B2_APPLICATION_KEY = os.getenv('B2_APPLICATION_KEY')
B2_BUCKET_INPUT = os.getenv('B2_BUCKET_INPUT', 'datos-entrada')
B2_BUCKET_OUTPUT = os.getenv('B2_BUCKET_OUTPUT', 'datos-salida')
B2_ENDPOINT = os.getenv('B2_ENDPOINT', 'https://s3.us-west-002.backblazeb2.com')

# Configuración de la aplicación
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = DATA_DIR / 'uploaded'
MODEL_DIR = BASE_DIR / 'models'
TEMPLATE_DIR = BASE_DIR / 'src' / 'templates'

# Crear directorios si no existen
for directory in [UPLOAD_DIR, MODEL_DIR, TEMPLATE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuración de correo
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
EMAIL_FROM = os.getenv('EMAIL_FROM', 'notificaciones@midominio.com')
