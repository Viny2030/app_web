import os
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
import json
from ..config import B2_BUCKET_OUTPUT
from .backblaze_service import BackblazeService

class DashboardService:
    """
    Servicio para generar y gestionar dashboards interactivos.
    """
    
    def __init__(self, b2_service: Optional[BackblazeService] = None):
        """
        Inicializa el servicio de dashboards.
        
        Args:
            b2_service: Instancia de BackblazeService para subir archivos
        """
        self.b2_service = b2_service or BackblazeService()
        self.bucket_name = B2_BUCKET_OUTPUT
    
    def generate_dashboard(
        self,
        df: pd.DataFrame,
        predictions: list,
        problem_type: str,
        results: Dict[str, Any],
        output_dir: str = "dashboards"
    ) -> str:
        """
        Genera un dashboard interactivo y devuelve la URL pública.
        
        Args:
            df: DataFrame con los datos
            predictions: Lista de predicciones del modelo
            problem_type: Tipo de problema (clasificación, regresión, clustering, etc.)
            results: Diccionario con métricas y resultados del modelo
            output_dir: Directorio local para guardar los archivos temporales
            
        Returns:
            str: URL pública del dashboard generado
        """
        # Crear directorio de salida
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_id = f"dashboard_{timestamp}_{uuid.uuid4().hex[:6]}"
        dashboard_dir = Path(output_dir) / dashboard_id
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar datos
        df_path = dashboard_dir / "data.csv"
        df.to_csv(df_path, index=False)
        
        # Crear archivo de configuración
        config = {
            "dashboard_id": dashboard_id,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat(),
            "problem_type": problem_type,
            "results": results,
            "data_columns": list(df.columns),
            "data_shape": {"rows": len(df), "columns": len(df.columns)}
        }
        
        with open(dashboard_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Crear archivo HTML estático
        self._create_static_dashboard(dashboard_dir, config)
        
        # Subir a Backblaze B2
        if self.b2_service:
            public_url = self._upload_to_backblaze(dashboard_dir, dashboard_id)
            return public_url
        
        return str(dashboard_dir)
    
    def _create_static_dashboard(self, dashboard_dir: Path, config: dict) -> None:
        """Crea un dashboard HTML estático con los datos."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard de Análisis - {config['dashboard_id']}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            <style>
                .dashboard-container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .metric-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            </style>
        </head>
        <body class="bg-gray-100">
            <div class="dashboard-container">
                <header class="mb-8">
                    <h1 class="text-3xl font-bold text-gray-800">📊 Dashboard de Análisis</h1>
                    <p class="text-gray-600">ID: {config['dashboard_id']}</p>
                </header>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                    <div class="metric-card">
                        <h3 class="text-lg font-semibold mb-2">📋 Información del Dataset</h3>
                        <p><span class="font-medium">Filas:</span> {config['data_shape']['rows']:,}</p>
                        <p><span class="font-medium">Columnas:</span> {config['data_shape']['columns']}</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3 class="text-lg font-semibold mb-2">🔍 Tipo de Análisis</h3>
                        <p class="capitalize">{config['problem_type'].replace('_', ' ')}</p>
                    </div>
                </div>
                
                <div class="bg-white rounded-lg shadow p-6 mb-8">
                    <h2 class="text-xl font-semibold mb-4">📈 Resultados del Modelo</h2>
                    <pre id="results-json" class="bg-gray-100 p-4 rounded overflow-auto"></pre>
                </div>
                
                <div id="plotly-chart" class="bg-white rounded-lg shadow p-6">
                    <h2 class="text-xl font-semibold mb-4">📊 Visualización de Datos</h2>
                    <div id="chart"></div>
                </div>
                
                <footer class="mt-12 text-center text-gray-500 text-sm">
                    <p>Dashboard generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Válido hasta: {(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}</p>
                </footer>
            </div>
            
            <script>
                // Mostrar resultados formateados
                const results = {json.dumps(config['results'], indent=2)};
                document.getElementById('results-json').textContent = JSON.stringify(results, null, 2);
                
                // Aquí podrías agregar lógica para cargar datos y generar gráficos con Plotly
                // Por ejemplo:
                // fetch('data.csv').then(...)
            </script>
        </body>
        </html>
        """
        
        with open(dashboard_dir / "index.html", "w", encoding="utf-8") as f:
            f.write(html_content)
    
    def _upload_to_backblaze(self, dashboard_dir: Path, dashboard_id: str) -> str:
        """
        Sube los archivos del dashboard a Backblaze B2 y devuelve la URL pública.
        """
        try:
            # Subir archivos individuales
            files_to_upload = [
                dashboard_dir / "index.html",
                dashboard_dir / "data.csv",
                dashboard_dir / "config.json"
            ]
            
            base_url = f"https://{self.bucket_name}.s3.{self.b2_service.endpoint_url.split('.')[-3]}.backblazeb2.com"
            uploaded_files = []
            
            for file_path in files_to_upload:
                if file_path.exists():
                    object_name = f"dashboards/{dashboard_id}/{file_path.name}"
                    public_url = self.b2_service.upload_file(
                        file_path=file_path,
                        bucket_name=self.bucket_name,
                        object_name=object_name
                    )
                    uploaded_files.append(public_url)
            
            # Devolver la URL del dashboard principal
            dashboard_url = f"{base_url}/dashboards/{dashboard_id}/index.html"
            return dashboard_url
            
        except Exception as e:
            print(f"Error al subir archivos a Backblaze B2: {e}")
            raise

    def get_dashboard_url(self, dashboard_id: str) -> str:
        """
        Obtiene la URL pública de un dashboard existente.
        
        Args:
            dashboard_id: ID del dashboard
            
        Returns:
            str: URL pública del dashboard
        """
        base_url = f"https://{self.bucket_name}.s3.{self.b2_service.endpoint_url.split('.')[-3]}.backblazeb2.com"
        return f"{base_url}/dashboards/{dashboard_id}/index.html"
