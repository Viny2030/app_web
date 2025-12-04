import os
import boto3
from botocore.client import Config


# b2_integrator.py

# ... (Código existente para subir_a_backblaze_b2)

# ==============================================================================
# 💡 NUEVA FUNCIÓN: Descargar y leer CSV de B2
# ==============================================================================
def descargar_y_leer_csv_b2(bucket_key):
    """
    Descarga un archivo CSV de Backblaze B2 y lo lee en un DataFrame.
    Utiliza las variables READ_KEY_ID, READ_APPLICATION_KEY, B2_BUCKET_NAME_RAW.
    """
    # 1. Obtener credenciales de lectura
    key_id = os.environ.get("READ_KEY_ID")
    application_key = os.environ.get("READ_APPLICATION_KEY")
    endpoint_url = os.environ.get("B2_ENDPOINT")
    bucket_name = os.environ.get("B2_BUCKET_NAME_RAW")

    if not all([key_id, application_key, endpoint_url, bucket_name]):
        print("❌ Error: Faltan variables de entorno de B2 (READ_*, B2_BUCKET_NAME_RAW, B2_ENDPOINT).")
        return None

    print(f"🌍 Descargando '{bucket_key}' de B2 Bucket: {bucket_name}...")

    try:
        # 2. Configurar la sesión de Boto3 para Backblaze S3
        s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=key_id,
            aws_secret_access_key=application_key,
            config=Config(signature_version='s3v4')
        )

        # 3. Descargar el archivo a un buffer en memoria
        obj = s3.get_object(Bucket=bucket_name, Key=bucket_key)

        # 4. Leer el CSV directamente desde el buffer
        df = pd.read_csv(obj['Body'])

        print("✅ Descarga y lectura exitosa.")
        return df

    except Exception as e:
        print(f"❌ ERROR al descargar de Backblaze B2: {e}")
        import traceback
        traceback.print_exc()
        return None