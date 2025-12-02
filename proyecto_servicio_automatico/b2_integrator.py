import boto3
import pandas as pd
import io
import os

# ==============================================================================
# 🎯 CONFIGURACIÓN Y CREDENCIALES DE BACKBLAZE B2
# ==============================================================================

# CLAVES DE LECTURA (READ): Usadas para descargar de 'dataset-raw'
# Clave nueva dedicada a lectura (Colab-Dataset-Raw)
READ_KEY_ID = '005a7f47ac71b1b0000000004'
READ_APPLICATION_KEY = 'K005Sy735QWuGZiqdoCTj2we6UHfIDU'

# CLAVES DE ESCRITURA (WRITE): Usadas para subir a 'repositorio-web'
# Clave anterior dedicada a escritura (Colab-repositorio-web)
WRITE_KEY_ID = '005a7f47ac71b1b0000000003'
WRITE_APPLICATION_KEY = 'K0051Lm50ayKiZ5X9PB96LpzaGfF2JM'

B2_BUCKET_NAME_RAW = 'dataset-raw'  # Bucket de entrada
B2_ENDPOINT_URL = 'https://s3.us-east-005.backblazeb2.com'


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