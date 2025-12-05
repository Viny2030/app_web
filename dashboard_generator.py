import streamlit as st
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from src.services.dashboard_service import DashboardService
from src.services.backblaze_service import BackblazeService

# Configuración de la página
st.set_page_config(
    page_title="Generador de Dashboards",
    page_icon="🚀",
    layout="wide"
)

def main():
    st.title("🚀 Generador de Dashboards")
    st.markdown("""
    Sube tus datos y genera un dashboard interactivo que podrás compartir con un enlace público.
    """)
    
    # Inicializar servicios
    try:
        b2_service = BackblazeService()
        dashboard_service = DashboardService(b2_service)
        st.session_state.b2_connected = True
    except Exception as e:
        st.error(f"No se pudo conectar con Backblaze B2: {str(e)}")
        st.warning("El dashboard se generará localmente pero no se podrá compartir.")
        st.session_state.b2_connected = False
    
    # Sección de carga de datos
    st.sidebar.header("1. Cargar Datos")
    uploaded_file = st.sidebar.file_uploader(
        "Sube tu archivo CSV o Excel",
        type=["csv", "xlsx", "xls"]
    )
    
    df = None
    if uploaded_file is not None:
        try:
            # Leer el archivo subido
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # Excel
                df = pd.read_excel(uploaded_file)
            
            st.sidebar.success("✅ Archivo cargado correctamente")
            
            # Mostrar vista previa
            with st.expander("🔍 Vista previa de los datos", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
                st.caption(f"Dimensión: {df.shape[0]} filas × {df.shape[1]} columnas")
                
                # Mostrar información de columnas
                st.subheader("📋 Información de columnas")
                col_info = pd.DataFrame({
                    'Columna': df.columns,
                    'Tipo': df.dtypes.astype(str),
                    'Valores únicos': df.nunique(),
                    'Valores nulos': df.isnull().sum()
                })
                st.dataframe(col_info, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error al leer el archivo: {str(e)}")
    
    # Sección de configuración del dashboard
    if df is not None:
        st.sidebar.header("2. Configurar Dashboard")
        
        # Seleccionar tipo de análisis
        problem_type = st.sidebar.selectbox(
            "Tipo de análisis",
            ["clasificacion", "regresion", "clustering", "analisis_exploratorio"],
            format_func=lambda x: {
                "clasificacion": "Clasificación",
                "regresion": "Regresión",
                "clustering": "Agrupamiento (Clustering)",
                "analisis_exploratorio": "Análisis Exploratorio"
            }[x]
        )
        
        # Generar resultados de ejemplo (en un caso real, estos vendrían de un modelo)
        results = {
            "tipo_analisis": problem_type,
            "fecha_generacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metricas": {
                "exactitud": 0.95 if problem_type == "clasificacion" else None,
                "rmse": 0.15 if problem_type == "regresion" else None,
                "silueta": 0.7 if problem_type == "clustering" else None,
                "clusters": 3 if problem_type == "clustering" else None
            },
            "columnas_utilizadas": list(df.columns[:min(5, len(df.columns))])
        }
        
        # Generar predicciones de ejemplo (en un caso real, estas vendrían de un modelo)
        predictions = ["Clase 1"] * len(df) if problem_type == "clasificacion" else None
        
        # Botón para generar el dashboard
        if st.sidebar.button("🚀 Generar Dashboard", use_container_width=True):
            with st.spinner("Generando dashboard..."):
                try:
                    # Crear directorio temporal
                    temp_dir = Path("temp_dashboards")
                    temp_dir.mkdir(exist_ok=True)
                    
                    # Generar el dashboard
                    dashboard_url = dashboard_service.generate_dashboard(
                        df=df,
                        predictions=predictions,
                        problem_type=problem_type,
                        results=results,
                        output_dir=str(temp_dir)
                    )
                    
                    # Mostrar el resultado
                    st.balloons()
                    st.success("¡Dashboard generado exitosamente!")
                    
                    # Mostrar enlace o botón de descarga
                    if dashboard_url.startswith("http"):
                        st.markdown(f"### 🌐 [Abrir Dashboard]({dashboard_url})")
                        st.code(dashboard_url, language="text")
                        
                        # Botón para copiar el enlace
                        st.button("📋 Copiar enlace", 
                                on_click=lambda: st.session_state.update({"copied": True}),
                                key="copy_button")
                        
                        if st.session_state.get("copied", False):
                            st.info("¡Enlace copiado al portapapeles!")
                    else:
                        st.warning("El dashboard se generó localmente pero no se pudo subir a la nube.")
                        st.info(f"Ruta local: {dashboard_url}")
                    
                except Exception as e:
                    st.error(f"Error al generar el dashboard: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()
