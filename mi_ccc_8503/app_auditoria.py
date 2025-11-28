# =================================================================
# SCRIPT DE INTERFAZ STREAMLIT PARA AUDITORÍA DE PRODUCTOS EN PROCESO
# Para ejecutar: streamlit run app_auditoria.py
# =================================================================

# --- 1. IMPORTACIONES ---

import streamlit as st
import pandas as pd
import boto3
import io
from botocore.client import Config
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CORRECCIÓN: Importar la función con el nombre correcto
from datos_auditoria import (
    generar_datos_simulados, 
    aplicar_auditoria_interactiva,
    analizar_correlaciones,
    analizar_eficiencia_etapa
)

# ----------------------------------------------------
## 1.1 CONFIGURACIÓN DE BACKBLAZE B2 (S3-Compatible)
# ----------------------------------------------------

# Credenciales de Backblaze B2 (desde secrets.toml)
try:
    B2_KEY_ID = st.secrets["b2"]["key_id"]
    B2_APPLICATION_KEY = st.secrets["b2"]["application_key"]
    B2_BUCKET_NAME = st.secrets["b2"]["bucket_name"]
    B2_ENDPOINT_URL_RAW = st.secrets["b2"]["endpoint_url_raw"]
except KeyError:
    st.error("⚠️ Error: Las credenciales de Backblaze B2 no están configuradas en secrets.toml")
    B2_KEY_ID = None
    B2_APPLICATION_KEY = None
    B2_BUCKET_NAME = 'dataset-raw'
    B2_ENDPOINT_URL_RAW = 's3.us-east-005.backblazeb2.com'

B2_ENDPOINT_URL = 'https://' + B2_ENDPOINT_URL_RAW


@st.cache_data
def load_data_from_b2(file_key='datos_simulados.csv'):
    """Carga un DataFrame desde Backblaze B2, con fallback a datos simulados."""
    # Verificar si las credenciales están disponibles
    if B2_KEY_ID is None or B2_APPLICATION_KEY is None:
        st.warning("⚠️ Credenciales de B2 no configuradas. Usando datos simulados.")
        return generar_datos_simulados()
    
    st.info(f"Intentando conectar a Backblaze B2 y cargar '{file_key}'...")
    try:
        s3 = boto3.client(
            service_name='s3',
            endpoint_url=B2_ENDPOINT_URL,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APPLICATION_KEY,
            config=Config(signature_version='s3v4')
        )

        obj = s3.get_object(Bucket=B2_BUCKET_NAME, Key=file_key)
        data = obj['Body'].read()

        df = pd.read_csv(io.BytesIO(data))
        st.success(f"✅ Datos cargados con éxito desde B2: {file_key}")
        return df

    except Exception as e:
        st.error(f"❌ Error al conectar o cargar datos desde Backblaze B2: {e}")
        st.warning("⚠️ Usando datos simulados localmente como fallback.")
        return generar_datos_simulados()

    # ===============================================================


# 2. CONFIGURACIÓN DE PÁGINA Y CACHÉ
# ===============================================================

st.set_page_config(
    page_title="Auditoría de Productos en Proceso", 
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
    <style>
    /* Estilos generales */
    .main {
        padding-top: 2rem;
    }
    
    /* Métricas mejoradas */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        color: #666;
    }
    
    /* Cards con sombra */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .metric-card-danger {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .metric-card-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .metric-card-warning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    /* Animaciones sutiles */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Headers mejorados */
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: #2c3e50;
        margin-top: 2rem;
    }
    
    /* Botones mejorados */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar mejorado */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Tabs mejorados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Widgets Interactivos ---
st.sidebar.title("🎛️ Parámetros de Auditoría")
st.sidebar.markdown("---")

st.sidebar.subheader("📊 Parámetros Principales")
umbral_min_avance = st.sidebar.slider(
    "Umbral Mínimo de Avance (%) para Producto A",
    0, 30, 10, help="Si el avance es menor a este valor, se genera una alerta."
)
cantidad_min_ensamblaje = st.sidebar.number_input(
    "Cantidad Mínima Requerida para Ensamblaje",
    10, 200, 50, help="Si la cantidad es menor a este valor en etapa Ensamblaje, se genera una alerta."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Configuración Avanzada")
mostrar_correlaciones = st.sidebar.checkbox("Mostrar Análisis de Correlaciones", value=True)
mostrar_clustering = st.sidebar.checkbox("Mostrar Análisis de Clustering", value=True)
mostrar_eficiencia = st.sidebar.checkbox("Mostrar Análisis de Eficiencia por Etapa", value=True)


@st.cache_data
def get_audit_data_interactive(min_avance, min_cantidad):
    """Carga datos (B2 o simulados) y aplica la auditoría con parámetros interactivos."""

    df_proceso = load_data_from_b2(file_key='datos_simulados.csv')

    # Llamamos a la función corregida
    df_auditado = aplicar_auditoria_interactiva(df_proceso, min_avance, min_cantidad)
    return df_auditado


# ===============================================================
# 3. INTERFAZ DE STREAMLIT
# ===============================================================

st.title("⚙️ Auditoría de Productos en Proceso (WIP)")
st.markdown(
    """
    <div class="fade-in">
    <p style="font-size: 1.1rem; color: #555;">
    Esta aplicación audita datos de productos en proceso, <strong>cargándolos desde Backblaze B2</strong> 
    (con <em>fallback</em> local). Utiliza un <strong>modelo de reglas interactivo</strong> y 
    <strong>algoritmos avanzados de detección de anomalías</strong> (ver barra lateral).
    </p>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- Muestra del Código Interactivo ---
with st.expander("📄 Ver Lógica de Auditoría Activa", expanded=False):
    st.code(f"""
# REGLA 1 (Avance Lento Producto A)
# Si el producto es 'Producto A' y el avance es menor a {umbral_min_avance}% y está activo:
# -> ALERTA: Avance lento.

# REGLA 2 (Avance 100% en etapa intermedia)
# Si el avance es >= 99.9% pero no está en etapa final:
# -> ALERTA: Avance al 100% en etapa intermedia

# REGLA 3 (Baja Cantidad Ensamblaje)
# Si la etapa es 'Ensamblaje' y la cantidad en proceso es menor a {cantidad_min_ensamblaje}:
# -> ALERTA: Baja cantidad para Ensamblaje.

# REGLA 4 (Procesos Estancados)
# Si avance < 30%, estado = Activo y cantidad > 100:
# -> ALERTA: Proceso posiblemente estancado

# REGLA 5 (Desviaciones por Línea)
# Si la línea se desvía > 20% del promedio global:
# -> ALERTA: Desviación significativa en línea

# REGLA 6 (Outliers Estadísticos)
# Detecta valores atípicos usando IQR (Interquartile Range)
# -> ALERTA: Valor atípico detectado
""", language='python')

st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
with col_btn1:
    iniciar_auditoria = st.button("🚀 Iniciar Auditoría", 
                                  help="Genera/Carga datos y aplica el análisis completo con los parámetros actuales.",
                                  use_container_width=True)

if iniciar_auditoria:
    with st.spinner('🔄 Ejecutando la auditoría con los parámetros seleccionados...'):
        # Pasa los valores de los widgets a la función de caché
        df_auditado = get_audit_data_interactive(umbral_min_avance, cantidad_min_ensamblaje)

        st.success("✅ Auditoría completada con éxito. Resultados ajustados a la lógica interactiva.")

        # --- Sección 1: Resumen y Alertas ---
        st.header("🔍 Resultados de la Auditoría")
        
        # Métricas mejoradas con KPIs visuales
        col1, col2, col3, col4 = st.columns(4)
        
        total_procesos = len(df_auditado)
        anomalias_count = len(df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo'])
        porcentaje_anomalias = (anomalias_count / total_procesos * 100) if total_procesos > 0 else 0
        score_riesgo_promedio = df_auditado['score_riesgo'].mean() if 'score_riesgo' in df_auditado.columns else 0
        
        with col1:
            st.metric(
                "📊 Total de Procesos", 
                f"{total_procesos:,}",
                delta=None
            )
        
        with col2:
            delta_color = "normal" if porcentaje_anomalias < 10 else "inverse"
            st.metric(
                "⚠️ Anomalías Detectadas", 
                f"{anomalias_count:,}",
                delta=f"{porcentaje_anomalias:.1f}%",
                delta_color=delta_color
            )
        
        with col3:
            riesgo_color = "normal" if score_riesgo_promedio < 30 else "inverse"
            st.metric(
                "🎯 Score de Riesgo Promedio", 
                f"{score_riesgo_promedio:.1f}",
                delta="0-100 escala",
                delta_color=riesgo_color
            )
        
        with col4:
            procesos_normales = total_procesos - anomalias_count
            st.metric(
                "✅ Procesos Normales", 
                f"{procesos_normales:,}",
                delta=f"{(procesos_normales/total_procesos*100):.1f}%",
                delta_color="normal"
            )

        # Organizar contenido en tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Detalles", "📈 Visualizaciones", "🔬 Análisis Avanzado", "📊 Estadísticas"])
        
        with tab1:
            st.subheader("📋 Procesos Anómalos o con Alertas")
            
            anomalies_and_alerts_df = df_auditado[
                (df_auditado['resultado_auditoria'] == 'Anómalo') | (df_auditado['alerta_heuristica'] != "Sin alertas")]
            
            if not anomalies_and_alerts_df.empty:
                st.info(f"**Total de procesos con alertas: {len(anomalies_and_alerts_df)}**")
                
                # Agregar score_riesgo si existe
                columnas_interes = ['id_proceso', 'producto', 'etapa_actual', 'linea_produccion', 
                                    'cantidad_en_proceso', 'avance_porcentaje', 'estado', 
                                    'alerta_heuristica', 'resultado_auditoria']
                if 'score_riesgo' in anomalies_and_alerts_df.columns:
                    columnas_interes.append('score_riesgo')
                if 'cluster' in anomalies_and_alerts_df.columns:
                    columnas_interes.append('cluster')
                
                # Ordenar por score de riesgo descendente
                if 'score_riesgo' in anomalies_and_alerts_df.columns:
                    anomalies_and_alerts_df = anomalies_and_alerts_df.sort_values('score_riesgo', ascending=False)
                
                st.dataframe(
                    anomalies_and_alerts_df[columnas_interes],
                    use_container_width=True,
                    height=400
                )

                csv_data = anomalies_and_alerts_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Reporte de Anomalías CSV",
                    data=csv_data,
                    file_name=f"reporte_anomalias_wip_{umbral_min_avance}_{cantidad_min_ensamblaje}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.success("✅ ¡No se encontraron anomalías o alertas significativas con los parámetros actuales!")
        
        with tab2:
            st.subheader("📈 Visualizaciones Interactivas")
            
            # Gráfico 1: Avance por Etapa (Box Plot con Plotly)
            fig1 = px.box(
                df_auditado, 
                x='avance_porcentaje', 
                y='etapa_actual', 
                color='resultado_auditoria',
                title='📊 Distribución de Avance por Etapa de Producción',
                labels={'avance_porcentaje': 'Avance (%)', 'etapa_actual': 'Etapa Actual'},
                color_discrete_map={'Normal': '#2ecc71', 'Anómalo': '#e74c3c'},
                orientation='h'
            )
            fig1.update_layout(
                height=500,
                showlegend=True,
                template='plotly_white',
                font=dict(size=12)
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Gráfico 2: Cantidad vs Avance (Scatter Plot)
            fig2 = px.scatter(
                df_auditado,
                x='cantidad_en_proceso',
                y='avance_porcentaje',
                color='resultado_auditoria',
                size='score_riesgo' if 'score_riesgo' in df_auditado.columns else None,
                symbol='linea_produccion',
                hover_data=['producto', 'etapa_actual', 'estado', 'alerta_heuristica'],
                title='📈 Cantidad en Proceso vs Avance (%)',
                labels={'cantidad_en_proceso': 'Cantidad en Proceso (unidades)', 
                       'avance_porcentaje': 'Avance (%)'},
                color_discrete_map={'Normal': '#3498db', 'Anómalo': '#e74c3c'}
            )
            fig2.update_layout(
                height=500,
                template='plotly_white',
                font=dict(size=12)
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Gráfico 3: Score de Riesgo por Producto
            if 'score_riesgo' in df_auditado.columns:
                fig3 = px.bar(
                    df_auditado.groupby('producto')['score_riesgo'].mean().reset_index(),
                    x='producto',
                    y='score_riesgo',
                    title='🎯 Score de Riesgo Promedio por Producto',
                    labels={'score_riesgo': 'Score de Riesgo', 'producto': 'Producto'},
                    color='score_riesgo',
                    color_continuous_scale='Reds'
                )
                fig3.update_layout(
                    height=400,
                    template='plotly_white',
                    font=dict(size=12)
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            # Gráfico 4: Clustering Visualization
            if 'cluster' in df_auditado.columns and mostrar_clustering:
                fig4 = px.scatter(
                    df_auditado,
                    x='cantidad_en_proceso',
                    y='avance_porcentaje',
                    color='cluster',
                    size='score_riesgo' if 'score_riesgo' in df_auditado.columns else None,
                    hover_data=['producto', 'etapa_actual', 'resultado_auditoria'],
                    title='🔍 Análisis de Clustering (Patrones Similares)',
                    labels={'cantidad_en_proceso': 'Cantidad en Proceso', 
                           'avance_porcentaje': 'Avance (%)'},
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig4.update_layout(
                    height=500,
                    template='plotly_white',
                    font=dict(size=12)
                )
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab3:
            st.subheader("🔬 Análisis Avanzado")
            
            col_anal1, col_anal2 = st.columns(2)
            
            with col_anal1:
                st.markdown("### 📊 Matriz de Correlaciones")
                if mostrar_correlaciones:
                    try:
                        corr_matrix = analizar_correlaciones(df_auditado)
                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            title="Correlaciones entre Variables Numéricas",
                            color_continuous_scale='RdBu',
                            labels=dict(color="Correlación")
                        )
                        fig_corr.update_layout(height=500)
                        st.plotly_chart(fig_corr, use_container_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo calcular la matriz de correlaciones: {e}")
            
            with col_anal2:
                st.markdown("### ⚡ Eficiencia por Etapa")
                if mostrar_eficiencia:
                    try:
                        eficiencia_df = analizar_eficiencia_etapa(df_auditado)
                        fig_eff = px.bar(
                            eficiencia_df.reset_index(),
                            x='etapa_actual',
                            y='eficiencia_score',
                            title='Eficiencia Score por Etapa',
                            labels={'eficiencia_score': 'Score de Eficiencia', 'etapa_actual': 'Etapa'},
                            color='eficiencia_score',
                            color_continuous_scale='Viridis'
                        )
                        fig_eff.update_layout(height=500, xaxis_tickangle=-45)
                        st.plotly_chart(fig_eff, use_container_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo calcular la eficiencia: {e}")
            
            # Distribución de alertas
            st.markdown("### 🚨 Distribución de Alertas")
            if 'alerta_heuristica' in df_auditado.columns:
                alertas_count = df_auditado[df_auditado['alerta_heuristica'] != 'Sin alertas']['alerta_heuristica'].value_counts()
                if len(alertas_count) > 0:
                    fig_alertas = px.pie(
                        values=alertas_count.values,
                        names=alertas_count.index,
                        title='Distribución de Tipos de Alertas',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_alertas.update_layout(height=400)
                    st.plotly_chart(fig_alertas, use_container_width=True)
                else:
                    st.info("No hay alertas activas en los datos actuales.")
        
        with tab4:
            st.subheader("📊 Estadísticas Descriptivas")
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.markdown("### 📈 Estadísticas por Variable")
                numeric_cols = df_auditado.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.dataframe(
                        df_auditado[numeric_cols].describe(),
                        use_container_width=True
                    )
            
            with col_stat2:
                st.markdown("### 📋 Resumen por Categoría")
                if 'producto' in df_auditado.columns:
                    resumen = df_auditado.groupby('producto').agg({
                        'id_proceso': 'count',
                        'avance_porcentaje': 'mean',
                        'cantidad_en_proceso': 'mean'
                    }).rename(columns={
                        'id_proceso': 'Total Procesos',
                        'avance_porcentaje': 'Avance Promedio (%)',
                        'cantidad_en_proceso': 'Cantidad Promedio'
                    })
                    st.dataframe(resumen, use_container_width=True)
            
            # Distribución de score de riesgo
            if 'score_riesgo' in df_auditado.columns:
                st.markdown("### 🎯 Distribución del Score de Riesgo")
                fig_hist = px.histogram(
                    df_auditado,
                    x='score_riesgo',
                    nbins=30,
                    title='Distribución del Score de Riesgo',
                    labels={'score_riesgo': 'Score de Riesgo', 'count': 'Frecuencia'},
                    color_discrete_sequence=['#3498db']
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)