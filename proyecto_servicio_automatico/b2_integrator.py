import boto3
import pandas as pd
import io
import os

# =============== VARIABLES DESDE ENTORNO ==================
READ_KEY_ID = os.getenv("READ_KEY_ID")
READ_APPLICATION_KEY = os.getenv("READ_APPLICATION_KEY")
WRITE_KEY_ID = os.getenv("WRITE_KEY_ID")
WRITE_APPLICATION_KEY = os.getenv("WRITE_APPLICATION_KEY")

B2_BUCKET_NAME_RAW = os.getenv("B2_BUCKET_NAME_RAW", "dataset-raw")
B2_BUCKET_RESULTADOS = os.getenv("B2_BUCKET_RESULTADOS", "repositorio-web")
B2_ENDPOINT_URL = os.getenv("B2_ENDPOINT_URL", "https://s3.us-east-005.backblazeb2.com")


# ==============================================================================
# FUNCIÓN DE DESCARGA (Usa las claves de LECTURA)
# ==============================================================================
def descargar_y_leer_csv_b2(nombre_archivo_b2: str) -> pd.DataFrame:
    """
    Descarga un archivo CSV de B2 usando las credenciales de LECTURA.
    """

    try:
        print(f"Iniciando conexión para DESCARGA en {B2_ENDPOINT_URL}...")

        # 1. Configurar la conexión S3 con las credenciales de LECTURA
        s3 = boto3.client(
            's3',
            endpoint_url=B2_ENDPOINT_URL,
            aws_access_key_id=READ_KEY_ID,
            aws_secret_access_key=READ_APPLICATION_KEY
        )

        # 2. Descargar el contenido del archivo
        print(f"-> Descargando el archivo '{nombre_archivo_b2}' del bucket '{B2_BUCKET_NAME_RAW}'...")

        response = s3.get_object(Bucket=B2_BUCKET_NAME_RAW, Key=nombre_archivo_b2)
        contenido_csv = response['Body'].read()

        # 3. Leer el contenido binario directamente en un DataFrame
        df = pd.read_csv(io.BytesIO(contenido_csv))

        print(f"-> Descarga y lectura exitosa. DataFrame cargado con {len(df)} filas y {len(df.columns)} columnas.")
        return df

    except Exception as e:
        print(f"❌ ERROR al interactuar con Backblaze B2 o leer CSV:")
        print(f"   Detalle: {e}")
        raise RuntimeError(
            "Fallo al descargar o procesar el archivo CSV desde Backblaze. (Verifique permisos de LECTURA en 'dataset-raw')") from e


# ==============================================================================
# FUNCIÓN DE SUBIDA (Usa las claves de ESCRITURA)
# ==============================================================================
def subir_archivo_b2(ruta_local: str, nombre_bucket: str, key_b2: str):
    """
    Sube un archivo local al bucket de Backblaze B2 especificado usando credenciales de ESCRITURA.
    """

    try:
        print(f"-> Subiendo '{ruta_local}' como '{key_b2}' al bucket '{nombre_bucket}'...")

        # 1. Configurar la conexión S3 con las credenciales de ESCRITURA
        s3 = boto3.client(
            's3',
            endpoint_url=B2_ENDPOINT_URL,
            aws_access_key_id=WRITE_KEY_ID,
            aws_secret_access_key=WRITE_APPLICATION_KEY
        )

        # 2. Subir el archivo
        s3.upload_file(
            Filename=ruta_local,
            Bucket=nombre_bucket,
            Key=key_b2
        )

        print("-> Subida completada.")

    except Exception as e:
        print(f"❌ ERROR al subir a Backblaze B2:")
        print(f"   Detalle: {e}")
        raise RuntimeError(
            "Fallo al subir el archivo al bucket de resultados de Backblaze. (Verifique permisos de ESCRITURA en 'repositorio-web')") from e