import os
import sys
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from src.services.backblaze_service import BackblazeService

# Cargar variables de entorno
load_dotenv()

def create_deployment_package(app_dir: str, output_dir: str = "deploy") -> str:
    """
    Crea un paquete de despliegue para la aplicación Streamlit.
    
    Args:
        app_dir: Directorio de la aplicación Streamlit
        output_dir: Directorio donde se creará el paquete
        
    Returns:
        str: Ruta al directorio del paquete de despliegue
    """
    app_dir = Path(app_dir).resolve()
    output_dir = Path(output_dir).resolve()
    
    # Crear directorio de despliegue
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Archivos necesarios para la aplicación
    required_files = [
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        '.env',
        'src/',
        'data/',
        'models/'
    ]
    
    print("📦 Creando paquete de despliegue...")
    
    # Copiar archivos necesarios
    for item in required_files:
        src = app_dir / item
        dst = output_dir / item
        
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"  ✓ Copiado archivo: {src}")
        elif src.is_dir() and src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  ✓ Copiado directorio: {src}")
    
    # Crear script de inicio personalizado
    with open(output_dir / 'start.sh', 'w') as f:
        f.write("""#!/bin/bash
# Script de inicio para la aplicación Streamlit

# Instalar dependencias
pip install -r requirements.txt

# Iniciar la aplicación
streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
""")
    
    print(f"✅ Paquete de despliegue creado en: {output_dir}")
    return str(output_dir)

def deploy_to_backblaze(deploy_dir: str, bucket_name: str = None) -> str:
    """
    Sube la aplicación empaquetada a Backblaze B2.
    
    Args:
        deploy_dir: Directorio con los archivos a desplegar
        bucket_name: Nombre del bucket en Backblaze B2
        
    Returns:
        str: URL pública de la aplicación desplegada
    """
    deploy_dir = Path(deploy_dir)
    if not deploy_dir.exists():
        raise FileNotFoundError(f"El directorio {deploy_dir} no existe")
    
    print("🚀 Desplegando aplicación a Backblaze B2...")
    
    # Inicializar el servicio de Backblaze
    b2_service = BackblazeService()
    
    # Subir directorio completo a B2
    uploaded_files = b2_service.upload_directory(
        directory_path=deploy_dir,
        bucket_name=bucket_name or os.getenv('B2_BUCKET_OUTPUT'),
        prefix='streamlit-app'
    )
    
    if not uploaded_files:
        raise RuntimeError("No se pudieron subir los archivos a Backblaze B2")
    
    # Obtener la URL base
    endpoint = os.getenv('B2_ENDPOINT', 'https://s3.us-west-002.backblazeb2.com')
    bucket_name = bucket_name or os.getenv('B2_BUCKET_OUTPUT')
    
    # Construir la URL de la aplicación
    app_url = f"{endpoint}/{bucket_name}/streamlit-app/index.html"
    
    print(f"✅ Aplicación desplegada exitosamente!")
    print(f"🌐 URL de la aplicación: {app_url}")
    
    return app_url

def main():
    """Función principal para desplegar la aplicación."""
    try:
        # Directorio de la aplicación
        app_dir = Path(__file__).parent
        
        # 1. Crear paquete de despliegue
        deploy_dir = create_deployment_package(app_dir)
        
        # 2. Desplegar a Backblaze B2
        app_url = deploy_to_backblaze(deploy_dir)
        
        print("\n🎉 ¡Despliegue completado exitosamente!")
        print(f"🔗 URL de la aplicación: {app_url}")
        
    except Exception as e:
        print(f"\n❌ Error durante el despliegue: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
