# ===============================================================
# SCRIPT DE AUDITORÍA DE INVERSIONES CON STREAMLIT Y DOCKER
# ===============================================================

# --- 1. IMPORTACIONES ---
import pandas as pd
# Importaciones necesarias (numpy, random, datetime, etc., deben estar en requirements.txt)
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st
import os # Importar os para manejar rutas

# --- RUTA DEL ARCHIVO DEL CLIENTE (VOLUMEN MONTADO) ---
# Esta ruta debe coincidir con la configuración del volumen en docker-compose.yml
CLIENT_DATA_PATH = '/app/data/dataset_inversiones_prueba.csv' 
# ===============================================================
# 2. FUNCIÓN DE CARGA DE DATOS (REEMPLAZA GENERACIÓN)
# ===============================================================

@st.cache_data
def cargar_dataset(file_path):
    """Carga el dataset CSV o Parquet desde el volumen montado."""
    if not os.path.exists(file_path):
        st.error(f"❌ Error: Archivo no encontrado en el contenedor. Ruta esperada: {file_path}")
        st.info("Asegúrese de que el archivo del cliente haya sido subido al VPS en /data/datasets/ y que docker-compose.yml esté bien configurado.")
        st.stop()
        
    try:
        # Simplificamos la carga asumiendo que el archivo subido es CSV (o adaptamos si el cliente sube Parquet/Excel)
        df = pd.read_csv(file_path)
        
        # CONVERSIÓN DE TIPOS CRÍTICA: Reasegurar que las fechas sean datetime
        df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'], errors='coerce')
        df['fecha_vencimiento'] = pd.to_datetime(df['fecha_vencimiento'], errors='coerce')
        
        return df

    except Exception as e:
        st.error(f"❌ Error al leer o procesar el dataset: {e}")
        st.stop()
        

# --- (TU FUNCIÓN generar_datos_simulados() HA SIDO ELIMINADA EN ESTA VERSIÓN) ---
# --- (TU FUNCIÓN aplicar_clustering() ES MANTENIDA) ---
def aplicar_clustering(df):
    """Aplica clustering K-Means para agrupar inversiones similares."""
    features = ['monto_inicial', 'tasa_anual_simulada', 'ganancia_perdida_simulada']
    X = df[features].copy ()

    # Escalar los datos para que K-Means funcione correctamente
    scaler = StandardScaler ()
    X_scaled = scaler.fit_transform (X.fillna (0))

    # Aplicar K-Means con 3 clusters
    kmeans = KMeans (n_clusters=3, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict (X_scaled)

    return df

# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_analisis(df):
    """Aplica la detección de anomalías y las reglas heurísticas."""
    # Las conversiones de fecha se movieron a la función cargar_dataset()
    
    # Detección de Anomalías (Isolation Forest)
    features = ['monto_inicial', 'tasa_anual_simulada', 'valor_actual_simulado', 'ganancia_perdida_simulada']
    X = df[features].copy ()
    scaler = StandardScaler ()
    X_scaled = scaler.fit_transform (X.fillna (0))
    iso_forest = IsolationForest (n_estimators=100, contamination=0.1, random_state=42)
    df['anomaly'] = iso_forest.fit_predict (X_scaled)
    df['resultado_auditoria'] = df['anomaly'].map ({1: 'Normal', -1: 'Anómalo'})

    # Reglas Heurísticas
    def auditoria_heuristica(row):
        alertas = []
        hoy = pd.to_datetime ('today')
        # Utilizamos try/except para manejar NaT (Not a Time) si la fecha no existe
        fecha_vencimiento = row['fecha_vencimiento']
        
        if pd.notnull(fecha_vencimiento) and fecha_vencimiento < hoy and row['estado_inversion'] != 'Liquidada':
            alertas.append ("Vencida no liquidada")
        if row['ganancia_perdida_simulada'] < 0:
            alertas.append ("Pérdida registrada")
        if row['tasa_anual_simulada'] < 0.02 or row['tasa_anual_simulada'] > 0.25:
            alertas.append ("Tasa fuera de rango")
        return " | ".join (alertas) if alertas else "Sin alertas"

    df['alerta_heuristica'] = df.apply (auditoria_heuristica, axis=1)

    # Aplicar el nuevo algoritmo de clustering
    df = aplicar_clustering (df)

    return df


# ===============================================================
# 4. INTERFAZ DE STREAMLIT
# ===============================================================

st.set_page_config(page_title="Auditoría de Inversiones", layout="wide")
st.title ("💰 Auditoría de Inversiones Temporarias")
st.markdown (
    "Esta aplicación realiza una auditoría de las inversiones cargadas, detectando anomalías con **Isolation Forest**, aplicando reglas heurísticas y agrupando datos con **K-Means**.")

# Usamos un botón de recarga simple ya que los datos se cargan desde el archivo
if st.button ("Ejecutar Auditoría", help="Carga el dataset del cliente y aplica el análisis"):
    with st.spinner ('Cargando datos y ejecutando la auditoría...'):
        # --- CAMBIO CRÍTICO: CARGAR DESDE ARCHIVO ---
        df = cargar_dataset(CLIENT_DATA_PATH)
        df_auditado = aplicar_analisis(df)
        
        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Resultados de la Auditoría")

        # ... (Resto de tu código para KPIs y tablas) ...
        col1, col2, col3 = st.columns (3)
        with col1:
            st.metric ("Total de Inversiones", len (df_auditado))
        with col2:
            anomalias_count = len (df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo'])
            st.metric ("Anomalías Detectadas", anomalias_count)
        with col3:
            clusters_count = df_auditado['cluster'].nunique ()
            st.metric ("Clusters Identificados", clusters_count)

        st.subheader ("Inversiones Anómalas y con Alertas")
        anomalies_df = df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo']
        if not anomalies_df.empty:
            columnas_interes = ['inversion_id', 'nombre_empresa', 'tipo_inversion', 'estado_inversion', 'monto_inicial',
                                'ganancia_perdida_simulada', 'tasa_anual_simulada', 'alerta_heuristica',
                                'resultado_auditoria']
            st.dataframe (anomalies_df[columnas_interes])

            csv_data = anomalies_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias.csv",
                mime="text/csv"
            )
        else:
            st.info ("¡No se encontraron anomalías significativas según el modelo!")

        # --- Sección 2: Visualizaciones Clave ---
        # ... (Resto de tu código de visualización) ...
        st.header ("📈 Visualizaciones")

        # Gráfico 1: Distribución de Tasas
        fig1, ax1 = plt.subplots (figsize=(10, 6))
        sns.histplot (df_auditado['tasa_anual_simulada'], kde=True, bins=15, color='skyblue', ax=ax1)
        ax1.set_title ('Distribución de Tasas Anuales Simuladas')
        ax1.set_xlabel ('Tasa Anual')
        ax1.set_ylabel ('Frecuencia')
        st.pyplot (fig1)

        # Gráfico 2: Anomalias en Ganancia vs Monto
        fig2, ax2 = plt.subplots (figsize=(12, 8))
        sns.scatterplot (
            data=df_auditado, x='monto_inicial', y='ganancia_perdida_simulada',
            hue='resultado_auditoria', style='estado_inversion', size='tasa_anual_simulada',
            sizes=(50, 250), palette={'Normal': 'green', 'Anómalo': 'red'}, alpha=0.7, ax=ax2
        )
        ax2.set_title ('Monto Inicial vs Ganancia/Pérdida (Detección de Anomalías)')
        ax2.set_xlabel ('Monto Inicial ($)')
        ax2.set_ylabel ('Ganancia / Pérdida Simulada ($)')
        ax2.get_xaxis ().set_major_formatter (plt.FuncFormatter (lambda x, p: format (int (x), ',')))
        ax2.get_yaxis ().set_major_formatter (plt.FuncFormatter (lambda y, p: format (int (y), ',')))
        ax2.legend (title='Resultado Auditoría')
        st.pyplot (fig2)

        # Gráfico 3: Matriz de Correlación
        st.subheader ("3. Matriz de Correlación")
        numerical_features = ['monto_inicial', 'valor_actual_simulado', 'ganancia_perdida_simulada',
                              'tasa_anual_simulada']
        corr_matrix = df_auditado[numerical_features].corr ()
        fig3, ax3 = plt.subplots (figsize=(10, 8))
        sns.heatmap (corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax3)
        ax3.set_title ('Matriz de Correlación de Variables Financieras')
        st.pyplot (fig3)

        # Gráfico 4: Clusters de Inversiones (K-Means)
        st.subheader ("4. Agrupación de Inversiones (K-Means)")
        fig4, ax4 = plt.subplots (figsize=(12, 8))
        sns.scatterplot (
            data=df_auditado, x='monto_inicial', y='ganancia_perdida_simulada',
            hue='cluster', palette='viridis', style='estado_inversion', s=100, ax=ax4
        )
        ax4.set_title ('Clustering de Inversiones por Monto y Ganancia')
        ax4.set_xlabel ('Monto Inicial ($)')
        ax4.set_ylabel ('Ganancia / Pérdida Simulada ($)')
        ax4.get_xaxis ().set_major_formatter (plt.FuncFormatter (lambda x, p: format (int (x), ',')))
        ax4.get_yaxis ().set_major_formatter (plt.FuncFormatter (lambda y, p: format (int (y), ',')))
        st.pyplot (fig4)

if __name__ == '__main__':
    st.set_page_config(page_title="Auditoría de Inversiones", layout="wide")
    st.title("🛡️ Dashboard de Auditoría de Inversiones Temporarias")
    st.markdown("""
        #### Estado: Esperando carga del Dataset del Cliente
        El archivo de datos debe estar disponible en la ruta: `/app/data/dataset_inversiones_prueba.csv`
    """)