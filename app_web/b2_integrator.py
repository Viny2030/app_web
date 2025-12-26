import os
import boto3
import io
import logging
from botocore.client import Config
from botocore.exceptions import ClientError
from typing import Optional, Dict, List, BinaryIO, Union
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class B2Client:
    """
    A client for interacting with Backblaze B2 storage service.
    Handles file uploads, downloads, and management with progress tracking.
    """
    
    def __init__(self, key_id: str = None, application_key: str = None, 
                 endpoint_url: str = None, bucket_name: str = None):
        """
        Initialize the B2 client with credentials.
        If no credentials are provided, they will be read from environment variables.
        """
        self.key_id = key_id or os.environ.get("B2_KEY_ID")
        self.application_key = application_key or os.environ.get("B2_APPLICATION_KEY")
        self.endpoint_url = endpoint_url or os.environ.get("B2_ENDPOINT")
        self.bucket_name = bucket_name or os.environ.get("B2_BUCKET_NAME")
        
        if not all([self.key_id, self.application_key, self.endpoint_url, self.bucket_name]):
            raise ValueError("Missing required B2 credentials. Please provide them or set environment variables.")
        
        self.s3_client = self._create_s3_client()
    
    def _create_s3_client(self):
        """Create and return a configured S3 client for Backblaze B2."""
        return boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.application_key,
            config=Config(
                signature_version='s3v4',
                retries={
                    'max_attempts': 3,
                    'mode': 'standard'
                }
            )
        )
    
    def upload_file(self, file_obj: BinaryIO, object_name: str, 
                   content_type: str = 'application/octet-stream', 
                   metadata: Optional[Dict] = None) -> bool:
        """
        Upload a file to Backblaze B2 with progress tracking.
        
        Args:
            file_obj: File-like object to upload
            object_name: S3 object name (key)
            content_type: MIME type of the file
            metadata: Optional metadata to store with the file
            
        Returns:
            bool: True if file was uploaded successfully, False otherwise
        """
        try:
            extra_args = {'ContentType': content_type}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                object_name,
                ExtraArgs=extra_args,
                Callback=ProgressPercentage(object_name, os.fstat(file_obj.fileno()).st_size)
            )
            logger.info(f"Successfully uploaded {object_name}")
            return True
            
        except ClientError as e:
            logger.error(f"Error uploading {object_name}: {e}")
            return False
    
    def download_file(self, object_name: str, file_path: str) -> bool:
        """
        Download a file from Backblaze B2 to a local file.
        
        Args:
            object_name: S3 object name (key)
            file_path: Local path to save the file to
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            with open(file_path, 'wb') as f:
                self.s3_client.download_fileobj(
                    self.bucket_name, 
                    object_name, 
                    f,
                    Callback=ProgressPercentage(object_name)
                )
            logger.info(f"Successfully downloaded {object_name} to {file_path}")
            return True
            
        except ClientError as e:
            logger.error(f"Error downloading {object_name}: {e}")
            return False
    
    def download_to_memory(self, object_name: str) -> Optional[bytes]:
        """
        Download a file from Backblaze B2 to memory.
        
        Args:
            object_name: S3 object name (key)
            
        Returns:
            bytes: File content if successful, None otherwise
        """
        try:
            buffer = io.BytesIO()
            self.s3_client.download_fileobj(
                self.bucket_name,
                object_name,
                buffer,
                Callback=ProgressPercentage(object_name)
            )
            buffer.seek(0)
            return buffer.read()
            
        except ClientError as e:
            logger.error(f"Error downloading {object_name}: {e}")
            return None
    
    def read_csv(self, object_name: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Read a CSV file from Backblaze B2 into a pandas DataFrame.
        
        Args:
            object_name: S3 object name (key)
            **kwargs: Additional arguments to pass to pandas.read_csv()
            
        Returns:
            pd.DataFrame: DataFrame containing the CSV data, or None if an error occurs
        """
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_name)
            return pd.read_csv(obj['Body'], **kwargs)
            
        except ClientError as e:
            logger.error(f"Error reading CSV {object_name}: {e}")
            return None
    
    def list_objects(self, prefix: str = '') -> List[Dict]:
        """
        List objects in the bucket with the given prefix.
        
        Args:
            prefix: Only list objects with this prefix
            
        Returns:
            List of dictionaries containing object metadata
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return response.get('Contents', [])
            
        except ClientError as e:
            logger.error(f"Error listing objects with prefix '{prefix}': {e}")
            return []
    
    def generate_presigned_url(self, object_name: str, 
                             expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL to share an S3 object

        Args:
            object_name: S3 object name (key)
            expiration: Time in seconds for the presigned URL to remain valid

        Returns:
            Presigned URL as string. If error, returns None.
        """
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_name
                },
                ExpiresIn=expiration
            )
            return response
            
        except ClientError as e:
            logger.error(f"Error generating presigned URL for {object_name}: {e}")
            return None
    
    def delete_object(self, object_name: str) -> bool:
        """
        Delete an object from the bucket.
        
        Args:
            object_name: S3 object name (key)
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_name)
            logger.info(f"Successfully deleted {object_name}")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting {object_name}: {e}")
            return False


class ProgressPercentage:
    """
    Callback class to display upload/download progress.
    """
    def __init__(self, filename: str, size: int = None):
        self._filename = filename
        self._size = size
        self._seen_so_far = 0
        self._last_percent = -1
    
    def __call__(self, bytes_amount):
        if self._size is None:
            # For downloads where we don't know the size in advance
            self._seen_so_far += bytes_amount
            logger.info(f"Downloading {self._filename}: {self._seen_so_far} bytes")
            return
            
        self._seen_so_far += bytes_amount
        percent = (self._seen_so_far / self._size) * 100
        
        # Only log when percentage changes to avoid log spam
        if int(percent) > self._last_percent:
            self._last_percent = int(percent)
            logger.info(f"Progress: {self._filename} - {self._seen_so_far}/{self._size} bytes ({percent:.1f}%)")


def get_b2_client() -> B2Client:
    """
    Factory function to get a configured B2Client instance using environment variables.
    """
    return B2Client()


def upload_file_to_b2(file_path: str, object_name: str = None) -> bool:
    """
    Helper function to upload a file to Backblaze B2.
    
    Args:
        file_path: Path to the local file to upload
        object_name: S3 object name (key). If not specified, the filename is used.
        
    Returns:
        bool: True if upload was successful, False otherwise
    """
    if not object_name:
        object_name = os.path.basename(file_path)
    
    try:
        b2_client = get_b2_client()
        with open(file_path, 'rb') as f:
            return b2_client.upload_file(f, object_name)
    except Exception as e:
        logger.error(f"Error in upload_file_to_b2: {e}")
        return False