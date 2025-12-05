import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from src.services.backblaze_service import BackblazeService

# Cargar variables de entorno
load_dotenv()

def convert_to_static(app_script: str, output_dir: str = "static_site") -> str:
    """
    Convierte una aplicación Streamlit en un sitio web estático.
    
    Args:
        app_script: Ruta al script principal de la aplicación Streamlit
        output_dir: Directorio donde se generará el sitio estático
        
    Returns:
        str: Ruta al directorio con el sitio estático generado
    """
    app_script = Path(app_script).resolve()
    output_dir = Path(output_dir).resolve()
    
    # Limpiar directorio de salida
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Convirtiendo aplicación a sitio estático...")
    
    try:
        # Instalar streamlit-static-export si no está instalado
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit-static-export"])
        
        # Configurar variables de entorno para streamlit
        env = os.environ.copy()
        env["STREAMLIT_SERVER_PORT"] = "8501"
        env["STREAMLIT_SERVER_HEADLESS"] = "true"
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        
        # Ejecutar streamlit con la opción de exportación estática
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "--server.port=8501", 
            "--server.headless=true", 
            "--browser.gatherUsageStats=false",
            "--server.fileWatcherType=none",
            "--server.runOnSave=false",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
            "--server.enableWebsocketCompression=false",
            "--server.enableStaticServing=true",
            "--server.staticFolder=\"" + str(output_dir) + "\"",
            str(app_script)
        ]
        
        # Ejecutar el comando
        print("🚀 Generando archivos estáticos...")
        subprocess.check_call(" ".join(cmd), shell=True, env=env)
        
        # Mover archivos generados a la carpeta de salida
        static_files = list(Path("./static").glob("*"))
        for file in static_files:
            shutil.move(str(file), output_dir / file.name)
        
        # Crear un index.html básico si no existe
        if not (output_dir / "index.html").exists():
            with open(output_dir / "index.html", "w") as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Mi Aplicación Streamlit</title>
                    <meta http-equiv="refresh" content="0; url='streamlit_app.html'" />
                </head>
                <body>
                    <p>Redirigiendo a la aplicación... <a href="streamlit_app.html">Haz clic aquí si no eres redirigido</a>.</p>
                </body>
                </html>
                """)
        
        print(f"✅ Sitio estático generado en: {output_dir}")
        return str(output_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al generar el sitio estático: {e}", file=sys.stderr)
        raise

def upload_static_site(static_dir: str, bucket_name: str = None) -> str:
    """
    Sube un sitio web estático a Backblaze B2.
    
    Args:
        static_dir: Directorio con los archivos estáticos
        bucket_name: Nombre del bucket en Backblaze B2
        
    Returns:
        str: URL pública del sitio web
    """
    static_dir = Path(static_dir)
    if not static_dir.exists():
        raise FileNotFoundError(f"El directorio {static_dir} no existe")
    
    print("☁️ Subiendo sitio web a Backblaze B2...")
    
    # Inicializar el servicio de Backblaze
    b2_service = BackblazeService()
    
    # Subir directorio completo a B2
    uploaded_files = b2_service.upload_directory(
        directory_path=static_dir,
        bucket_name=bucket_name or os.getenv('B2_BUCKET_OUTPUT'),
        prefix='streamlit-app'
    )
    
    if not uploaded_files:
        raise RuntimeError("No se pudieron subir los archivos a Backblaze B2")
    
    # Obtener la URL base
    endpoint = os.getenv('B2_ENDPOINT', 'https://f004.backblazeb2.com/file')
    bucket_name = bucket_name or os.getenv('B2_BUCKET_OUTPUT')
    
    # Construir la URL de la aplicación
    app_url = f"{endpoint}/{bucket_name}/streamlit-app/index.html"
    
    print(f"✅ Sitio web subido exitosamente!")
    print(f"🌐 URL del sitio: {app_url}")
    
    return app_url

def main():
    """Función principal para desplegar la aplicación."""
    try:
        # Directorio de la aplicación
        app_dir = Path(__file__).parent
        
        # 1. Convertir a sitio estático
        static_dir = convert_to_static(
            app_script=app_dir / "dashboard_generator.py",
            output_dir=app_dir / "static_site"
        )
        
        # 2. Subir a Backblaze B2
        app_url = upload_static_site(static_dir)
        
        print("\n🎉 ¡Despliegue completado exitosamente!")
        print(f"🔗 URL de la aplicación: {app_url}")
        
    except Exception as e:
        print(f"\n❌ Error durante el despliegue: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
