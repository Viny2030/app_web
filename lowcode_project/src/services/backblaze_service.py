import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Optional, Union
import logging
from ..config import B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET_INPUT, B2_BUCKET_OUTPUT, B2_ENDPOINT

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackblazeService:
    """Servicio para interactuar con Backblaze B2"""
    
    def __init__(self, key_id: str = None, application_key: str = None):
        """Inicializa el cliente de Backblaze B2"""
        self.key_id = key_id or B2_KEY_ID
        self.application_key = application_key or B2_APPLICATION_KEY
        self.endpoint_url = B2_ENDPOINT
        self.client = self._create_client()
    
    def _create_client(self):
        """Crea y retorna un cliente de boto3 configurado para Backblaze B2"""
        if not self.key_id or not self.application_key:
            raise ValueError("Se requieren key_id y application_key para autenticación con Backblaze B2")
        
        return boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.application_key
        )
    
    def upload_file(self, file_path: Union[str, Path], bucket_name: str = None, 
                   object_name: Optional[str] = None) -> str:
        """
        Sube un archivo a un bucket de Backblaze B2
        
        Args:
            file_path: Ruta local al archivo a subir
            bucket_name: Nombre del bucket (por defecto usa B2_BUCKET_INPUT)
            object_name: Nombre del objeto en S3 (si no se especifica, usa el nombre del archivo)
            
        Returns:
            str: URL pública del archivo subido
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo {file_path} no existe")
            
        bucket_name = bucket_name or B2_BUCKET_INPUT
        object_name = object_name or file_path.name
        
        try:
            self.client.upload_file(
                str(file_path),
                bucket_name,
                object_name,
                ExtraArgs={
                    'ACL': 'public-read',
                    'ContentType': self._get_content_type(file_path)
                }
            )
            
            # Construir URL pública
            public_url = f"{self.endpoint_url}/{bucket_name}/{object_name}"
            logger.info(f"Archivo subido exitosamente a {public_url}")
            return public_url
            
        except ClientError as e:
            logger.error(f"Error al subir el archivo a Backblaze B2: {e}")
            raise
    
    def download_file(self, object_name: str, local_path: Union[str, Path], 
                     bucket_name: str = None) -> Path:
        """
        Descarga un archivo desde Backblaze B2
        
        Args:
            object_name: Nombre del objeto en S3
            local_path: Ruta local donde guardar el archivo
            bucket_name: Nombre del bucket (por defecto usa B2_BUCKET_INPUT)
            
        Returns:
            Path: Ruta al archivo descargado
        """
        if isinstance(local_path, str):
            local_path = Path(local_path)
            
        local_path.parent.mkdir(parents=True, exist_ok=True)
        bucket_name = bucket_name or B2_BUCKET_INPUT
        
        try:
            self.client.download_file(bucket_name, object_name, str(local_path))
            logger.info(f"Archivo descargado exitosamente a {local_path}")
            return local_path
            
        except ClientError as e:
            logger.error(f"Error al descargar el archivo desde Backblaze B2: {e}")
            raise
    
    def list_objects(self, bucket_name: str = None, prefix: str = '') -> list:
        """
        Lista los objetos en un bucket
        
        Args:
            bucket_name: Nombre del bucket (por defecto usa B2_BUCKET_INPUT)
            prefix: Prefijo para filtrar objetos
            
        Returns:
            list: Lista de objetos en el bucket
        """
        bucket_name = bucket_name or B2_BUCKET_INPUT
        
        try:
            response = self.client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            return response.get('Contents', [])
            
        except ClientError as e:
            logger.error(f"Error al listar objetos en el bucket {bucket_name}: {e}")
            raise
    
    @staticmethod
    def _get_content_type(file_path: Path) -> str:
        """Obtiene el tipo MIME de un archivo basado en su extensión"""
        content_types = {
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.pdf': 'application/pdf',
            '.zip': 'application/zip'
        }
        
        ext = file_path.suffix.lower()
        return content_types.get(ext, 'application/octet-stream')
