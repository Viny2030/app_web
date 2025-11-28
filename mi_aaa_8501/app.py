import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
import boto3
from io import BytesIO
import time

# ===============================================================
# SCRIPT DE AUDITORÍA DE INVERSIONES CON STREAMLIT Y B2
# ===============================================================

warnings.filterwarnings("ignore", category=ConvergenceWarning)

st.set_page_config(
    page_title="Auditoría de Inversiones - Dashboard Completo",
    layout="wide",
    page_icon="💰"
)

# --- PARÁMETROS DE B2 CLOUD STORAGE ---
import os

# Intentar obtener las credenciales de variables de entorno (para producción)
B2_KEY_ID = os.environ.get('B2_KEY_ID')
B2_APPLICATION_KEY = os.environ.get('B2_APPLICATION_KEY')
B2_BUCKET_NAME = os.environ.get('B2_BUCKET_NAME')
B2_ENDPOINT_URL_RAW = os.environ.get('B2_ENDPOINT_URL_RAW')

# Si no están en las variables de entorno, intentar con secrets.toml (para desarrollo local)
if not all([B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET_NAME, B2_ENDPOINT_URL_RAW]):
    try:
        B2_KEY_ID = st.secrets["b2"]["key_id"]
        B2_APPLICATION_KEY = st.secrets["b2"]["application_key"]
        B2_BUCKET_NAME = st.secrets["b2"]["bucket_name"]
        B2_ENDPOINT_URL_RAW = st.secrets["b2"]["endpoint_url_raw"]
    except (KeyError, FileNotFoundError):
        st.error("⚠️ Error: Las credenciales de Backblaze B2 no están configuradas correctamente.")
        st.info("Por favor, configura las siguientes variables de entorno:")
        st.code("""
        B2_KEY_ID=tu_key_id
        B2_APPLICATION_KEY=tu_application_key
        B2_BUCKET_NAME=tu_bucket_name
        B2_ENDPOINT_URL_RAW=tu_endpoint_url
        """)
        st.stop()

B2_ENDPOINT_URL = 'https://' + B2_ENDPOINT_URL_RAW if B2_ENDPOINT_URL_RAW else None
FILE_KEY_IN_B2 = 'dataset_inversiones_prueba.csv'
DEFAULT_N_CLUSTERS = 4

# ---------------------------------------------------------------
# FUNCIÓN DE LECTURA DE DATOS DESDE BACKBLAZE B2
# ---------------------------------------------------------------

@st.cache_data(show_spinner="Cargando datos desde Backblaze B2...")
def load_data_from_b2():
    """Carga todos los datos desde B2."""
    if not all([B2_ENDPOINT_URL, B2_KEY_ID, B2_APPLICATION_KEY]):
        return pd.DataFrame()
    
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=B2_ENDPOINT_URL,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APPLICATION_KEY
        )
        obj = s3.get_object(Bucket=B2_BUCKET_NAME, Key=FILE_KEY_IN_B2)
        csv_data = obj['Body'].read()
        df = pd.read_csv(BytesIO(csv_data))
        df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'], errors='coerce')
        df['fecha_vencimiento'] = pd.to_datetime(df['fecha_vencimiento'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo '{FILE_KEY_IN_B2}' desde Backblaze B2.")
        st.code(f"Detalle del Error: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------
# FUNCIONES DE ANÁLISIS
# ---------------------------------------------------------------

def apply_analysis(df, n_clusters):
    """Prepara y aplica K-Means para agrupar las inversiones."""
    df_auditado = df.copy()

    features = [
        'monto_inicial',
        'tasa_anual_simulada',
        'valor_actual_simulado',
        'ganancia_perdida_simulada'
    ]

    df_auditado['dias_plazo_flag'] = df_auditado['dias_plazo'].apply(lambda x: 1 if pd.notna(x) else 0)
    df_auditado['dias_plazo'] = df_auditado['dias_plazo'].fillna(0)

    features.append('dias_plazo')
    features.append('dias_plazo_flag')

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_auditado[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df_auditado['cluster'] = kmeans.fit_predict(data_scaled)
    inertia = kmeans.inertia_

    try:
        silhouette = silhouette_score(data_scaled, df_auditado['cluster'])
        davies_bouldin = davies_bouldin_score(data_scaled, df_auditado['cluster'])
    except:
        silhouette = 0
        davies_bouldin = 0

    return df_auditado, inertia, silhouette, davies_bouldin, scaler, features, data_scaled, kmeans


def calculate_elbow_method(df, max_k=10, features_list=None):
    """Calcula el método del codo para determinar el número óptimo de clusters."""
    if features_list is None:
        return [], []
    
    df_temp = df.copy()
    df_temp['dias_plazo_flag'] = df_temp['dias_plazo'].apply(lambda x: 1 if pd.notna(x) else 0)
    df_temp['dias_plazo'] = df_temp['dias_plazo'].fillna(0)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_temp[features_list])
    
    inertias = []
    silhouettes = []
    k_range = range(2, min(max_k + 1, len(df_temp)))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(data_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(data_scaled, labels))
    
    return list(k_range), inertias, silhouettes


def apply_pca(data_scaled, n_components=3):
    """Aplica PCA para reducción de dimensionalidad."""
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)
    return pca_result, pca.explained_variance_ratio_


# ---------------------------------------------------------------
# CARGAR DATOS
# ---------------------------------------------------------------

df_inversiones = load_data_from_b2()

if df_inversiones.empty:
    st.error("❌ No se pudieron cargar los datos. Verifica la conexión con Backblaze B2.")
    st.stop()

# ---------------------------------------------------------------
# SIDEBAR - PARÁMETROS DE ANÁLISIS
# ---------------------------------------------------------------

st.sidebar.header("⚙️ Parámetros de Análisis")

# Filtro opcional por empresa
empresas_disponibles = ['Todas'] + list(df_inversiones['nombre_empresa'].unique())
empresa_seleccionada = st.sidebar.selectbox(
    "Filtrar por Empresa",
    empresas_disponibles,
    index=0,
    help="Selecciona una empresa específica o 'Todas' para ver todo el dataset"
)

if empresa_seleccionada != 'Todas':
    df_inversiones = df_inversiones[df_inversiones['nombre_empresa'] == empresa_seleccionada].copy()

# Filtro por tipo de inversión
tipos_disponibles = ['Todos'] + list(df_inversiones['tipo_inversion'].unique())
tipo_seleccionado = st.sidebar.selectbox(
    "Filtrar por Tipo de Inversión",
    tipos_disponibles,
    index=0
)

if tipo_seleccionado != 'Todos':
    df_inversiones = df_inversiones[df_inversiones['tipo_inversion'] == tipo_seleccionado].copy()

# Filtro por estado
estados_disponibles = ['Todos'] + list(df_inversiones['estado_inversion'].unique())
estado_seleccionado = st.sidebar.selectbox(
    "Filtrar por Estado",
    estados_disponibles,
    index=0
)

if estado_seleccionado != 'Todos':
    df_inversiones = df_inversiones[df_inversiones['estado_inversion'] == estado_seleccionado].copy()

# Validación de datos suficientes
if len(df_inversiones) < 2:
    st.warning("⚠️ Los filtros seleccionados devuelven menos de 2 registros. Por favor ajusta los filtros.")
    st.info(f"Registros encontrados: {len(df_inversiones)}")
    if len(df_inversiones) == 1:
        st.dataframe(df_inversiones)
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info(f"📊 Registros: {len(df_inversiones):,}")

max_clusters = min(10, len(df_inversiones) - 1)
N_CLUSTERS = st.sidebar.slider(
    "Número de Clusters (Grupos)",
    min_value=2,
    max_value=max(2, max_clusters),
    value=min(DEFAULT_N_CLUSTERS, max(2, max_clusters)),
    step=1,
    help="Define cuántos grupos de similaridad debe buscar el algoritmo K-Means."
)

st.sidebar.markdown("---")
st.sidebar.header("🎨 Opciones de Visualización")
show_elbow = st.sidebar.checkbox("Mostrar Método del Codo", value=True)
show_3d = st.sidebar.checkbox("Visualización 3D", value=True)
show_pca = st.sidebar.checkbox("Análisis PCA", value=True)
show_distributions = st.sidebar.checkbox("Distribuciones y Análisis", value=True)
show_time_series = st.sidebar.checkbox("Análisis Temporal", value=True)
show_empresa_analysis = st.sidebar.checkbox("Análisis por Empresa", value=True)

# ---------------------------------------------------------------
# APLICAR ANÁLISIS
# ---------------------------------------------------------------

with st.spinner("🔄 Ejecutando análisis K-Means..."):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 30:
            status_text.text("📊 Preparando datos...")
        elif i < 60:
            status_text.text("🔍 Escalando características...")
        elif i < 90:
            status_text.text("🎯 Aplicando clustering K-Means...")
        else:
            status_text.text("✨ Calculando métricas...")
        time.sleep(0.005)
    
    df_auditado, kmeans_inertia, silhouette, davies_bouldin, scaler, features_list, data_scaled, kmeans_model = apply_analysis(df_inversiones, N_CLUSTERS)
    
    progress_bar.progress(100)
    status_text.text("✅ Análisis completado!")
    time.sleep(0.3)
    progress_bar.empty()
    status_text.empty()

# ---------------------------------------------------------------
# TÍTULO Y MÉTRICAS PRINCIPALES
# ---------------------------------------------------------------

st.title("💰 Dashboard de Auditoría de Inversiones")
st.markdown(f"**Total de registros analizados:** `{len(df_auditado):,}`")
st.markdown("---")

# Métricas principales
total_inversiones = df_auditado['monto_inicial'].sum()
ganancia_neta = df_auditado['ganancia_perdida_simulada'].sum()
num_activas = df_auditado[df_auditado['estado_inversion'] == 'Activa'].shape[0]
avg_tasa = df_auditado['tasa_anual_simulada'].mean()
num_empresas = df_auditado['nombre_empresa'].nunique()
num_tipos = df_auditado['tipo_inversion'].nunique()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("💵 Monto Total", f"${total_inversiones:,.0f}")
col2.metric("📈 Ganancia Neta", f"${ganancia_neta:,.0f}", delta=f"{ganancia_neta / total_inversiones * 100:.2f}%")
col3.metric("✅ Inversiones Activas", f"{num_activas:,}")
col4.metric("📊 Tasa Promedio", f"{avg_tasa:.2%}")
col5.metric("🏢 Empresas", f"{num_empresas}")
col6.metric("📋 Tipos de Inversión", f"{num_tipos}")

# Métricas del modelo
with st.expander("🛠️ Métricas del Modelo K-Means"):
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    
    col_met1.metric("Inertia (WCSS)", f"{kmeans_inertia:,.2f}")
    col_met2.metric("Silhouette Score", f"{silhouette:.3f}")
    col_met3.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}")
    col_met4.metric("Features Utilizadas", f"{len(features_list)}")
    
    st.markdown(f"**Variables Usadas:** `{', '.join(features_list)}`")
    
    cluster_summary = df_auditado.groupby('cluster').agg({
        'monto_inicial': ['count', 'sum', 'mean'],
        'ganancia_perdida_simulada': ['sum', 'mean'],
        'tasa_anual_simulada': 'mean'
    }).round(2)
    cluster_summary.columns = ['Cantidad', 'Monto_Total', 'Monto_Promedio', 'Ganancia_Total', 'Ganancia_Promedio', 'Tasa_Promedio']
    st.dataframe(cluster_summary, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------
# TABS DE VISUALIZACIÓN
# ---------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 Clustering", 
    "🏢 Por Empresa",
    "📈 Rendimiento", 
    "📊 Distribuciones",
    "⏱️ Temporal",
    "🔬 Análisis Avanzado",
    "📋 Datos"
])

# ---------------------------------------------------------------
# TAB 1: CLUSTERING
# ---------------------------------------------------------------
with tab1:
    st.subheader("🎯 Análisis de Clustering K-Means")
    st.markdown(f"Las inversiones se han agrupado en **{N_CLUSTERS} clusters** por similaridad financiera.")

    if show_elbow:
        st.markdown("### 📉 Método del Codo y Silhouette")
        with st.spinner("Calculando..."):
            k_range, inertias, silhouettes = calculate_elbow_method(df_inversiones, max_k=min(10, len(df_inversiones)), features_list=features_list)
        
        if k_range and inertias:
            col_elbow1, col_elbow2 = st.columns(2)
            
            with col_elbow1:
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(x=k_range, y=inertias, mode='lines+markers', name='Inertia', line=dict(color='#1f77b4', width=3)))
                if N_CLUSTERS in k_range:
                    idx = k_range.index(N_CLUSTERS)
                    fig_elbow.add_trace(go.Scatter(x=[N_CLUSTERS], y=[inertias[idx]], mode='markers', name=f'K={N_CLUSTERS}', marker=dict(size=15, color='red', symbol='star')))
                fig_elbow.update_layout(title='Método del Codo', xaxis_title='K', yaxis_title='Inertia', template='plotly_white')
                st.plotly_chart(fig_elbow, use_container_width=True)
            
            with col_elbow2:
                fig_sil = go.Figure()
                fig_sil.add_trace(go.Scatter(x=k_range, y=silhouettes, mode='lines+markers', name='Silhouette', line=dict(color='#2ca02c', width=3)))
                if N_CLUSTERS in k_range:
                    idx = k_range.index(N_CLUSTERS)
                    fig_sil.add_trace(go.Scatter(x=[N_CLUSTERS], y=[silhouettes[idx]], mode='markers', name=f'K={N_CLUSTERS}', marker=dict(size=15, color='red', symbol='star')))
                fig_sil.update_layout(title='Silhouette Score por K', xaxis_title='K', yaxis_title='Silhouette', template='plotly_white')
                st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("### 🎨 Visualización 2D de Clusters")
    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        x_axis = st.selectbox("Eje X", ['monto_inicial', 'valor_actual_simulado', 'tasa_anual_simulada', 'ganancia_perdida_simulada', 'dias_plazo'], index=0)
    with col_viz2:
        y_axis = st.selectbox("Eje Y", ['monto_inicial', 'valor_actual_simulado', 'tasa_anual_simulada', 'ganancia_perdida_simulada', 'dias_plazo'], index=3)
    
    fig1 = px.scatter(
        df_auditado, x=x_axis, y=y_axis, color='cluster', symbol='estado_inversion',
        size='monto_inicial', hover_data=['nombre_empresa', 'tipo_inversion', 'tasa_anual_simulada'],
        title=f'Clustering: {x_axis} vs {y_axis}', color_continuous_scale='viridis'
    )
    st.plotly_chart(fig1, use_container_width=True)

    if show_3d:
        st.markdown("### 🌐 Visualización 3D")
        col_3d1, col_3d2, col_3d3 = st.columns(3)
        with col_3d1:
            x_3d = st.selectbox("Eje X (3D)", ['monto_inicial', 'valor_actual_simulado', 'tasa_anual_simulada'], index=0, key='x3d')
        with col_3d2:
            y_3d = st.selectbox("Eje Y (3D)", ['ganancia_perdida_simulada', 'tasa_anual_simulada', 'dias_plazo'], index=0, key='y3d')
        with col_3d3:
            z_3d = st.selectbox("Eje Z (3D)", ['tasa_anual_simulada', 'dias_plazo', 'valor_actual_simulado'], index=0, key='z3d')
        
        fig_3d = px.scatter_3d(
            df_auditado, x=x_3d, y=y_3d, z=z_3d, color='cluster',
            symbol='estado_inversion', size='monto_inicial',
            hover_data=['nombre_empresa', 'tipo_inversion'],
            title=f'Visualización 3D (K={N_CLUSTERS})', color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    if show_pca:
        st.markdown("### 🔬 Análisis PCA (Reducción de Dimensionalidad)")
        pca_result, variance_ratio = apply_pca(data_scaled, n_components=3)
        df_auditado['PCA1'] = pca_result[:, 0]
        df_auditado['PCA2'] = pca_result[:, 1]
        df_auditado['PCA3'] = pca_result[:, 2]
        
        col_pca1, col_pca2 = st.columns(2)
        with col_pca1:
            fig_pca2d = px.scatter(
                df_auditado, x='PCA1', y='PCA2', color='cluster',
                hover_data=['nombre_empresa', 'monto_inicial'],
                title=f'PCA 2D (Varianza: {sum(variance_ratio[:2])*100:.1f}%)'
            )
            st.plotly_chart(fig_pca2d, use_container_width=True)
        
        with col_pca2:
            fig_var = px.bar(
                x=[f'PC{i+1}' for i in range(len(variance_ratio))],
                y=variance_ratio * 100,
                title='Varianza Explicada por Componente',
                labels={'x': 'Componente', 'y': 'Varianza (%)'}
            )
            st.plotly_chart(fig_var, use_container_width=True)

    st.markdown("### 🔗 Matriz de Correlación")
    numerical_features = ['monto_inicial', 'valor_actual_simulado', 'ganancia_perdida_simulada', 'tasa_anual_simulada', 'dias_plazo']
    corr_matrix = df_auditado[numerical_features].corr()
    fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu', title='Correlación de Variables')
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------------
# TAB 2: ANÁLISIS POR EMPRESA
# ---------------------------------------------------------------
with tab2:
    if show_empresa_analysis:
        st.subheader("🏢 Análisis por Empresa")
        
        df_empresa = df_auditado.groupby('nombre_empresa').agg({
            'monto_inicial': ['sum', 'mean', 'count'],
            'ganancia_perdida_simulada': ['sum', 'mean'],
            'tasa_anual_simulada': 'mean',
            'cluster': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        }).round(2)
        df_empresa.columns = ['Monto_Total', 'Monto_Promedio', 'Cantidad', 'Ganancia_Total', 'Ganancia_Promedio', 'Tasa_Promedio', 'Cluster_Predominante']
        df_empresa['ROI_%'] = (df_empresa['Ganancia_Total'] / df_empresa['Monto_Total'] * 100).round(2)
        df_empresa = df_empresa.reset_index().sort_values('Monto_Total', ascending=False)
        
        col_emp1, col_emp2 = st.columns(2)
        
        with col_emp1:
            fig_emp_monto = px.bar(
                df_empresa.head(15), x='nombre_empresa', y='Monto_Total',
                color='ROI_%', title='Top 15 Empresas por Monto Total',
                color_continuous_scale='Greens'
            )
            fig_emp_monto.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_emp_monto, use_container_width=True)
        
        with col_emp2:
            fig_emp_ganancia = px.bar(
                df_empresa.head(15), x='nombre_empresa', y='Ganancia_Total',
                color='Tasa_Promedio', title='Top 15 Empresas por Ganancia',
                color_continuous_scale='Blues'
            )
            fig_emp_ganancia.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_emp_ganancia, use_container_width=True)
        
        fig_treemap = px.treemap(
            df_auditado, path=['nombre_empresa', 'tipo_inversion'], values='monto_inicial',
            color='ganancia_perdida_simulada', color_continuous_scale='RdYlGn',
            title='Treemap: Monto por Empresa y Tipo de Inversión'
        )
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        fig_sunburst = px.sunburst(
            df_auditado, path=['nombre_empresa', 'tipo_inversion', 'estado_inversion'],
            values='monto_inicial', color='ganancia_perdida_simulada',
            color_continuous_scale='RdYlGn', title='Sunburst: Estructura de Inversiones'
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        st.markdown("### 📋 Tabla de Empresas")
        st.dataframe(df_empresa, use_container_width=True)

# ---------------------------------------------------------------
# TAB 3: RENDIMIENTO
# ---------------------------------------------------------------
with tab3:
    st.subheader("📈 Análisis de Rendimiento")
    
    df_rendimiento = df_auditado.groupby('tipo_inversion').agg({
        'monto_inicial': ['sum', 'mean', 'count'],
        'ganancia_perdida_simulada': ['sum', 'mean'],
        'tasa_anual_simulada': 'mean'
    }).round(2)
    df_rendimiento.columns = ['Monto_Total', 'Monto_Promedio', 'Cantidad', 'Ganancia_Total', 'Ganancia_Promedio', 'Tasa_Promedio']
    df_rendimiento['ROI_%'] = (df_rendimiento['Ganancia_Total'] / df_rendimiento['Monto_Total'] * 100).round(2)
    df_rendimiento = df_rendimiento.reset_index()

    col_rend1, col_rend2 = st.columns(2)
    
    with col_rend1:
        fig2 = px.bar(
            df_rendimiento, x='tipo_inversion', y='Ganancia_Promedio',
            color='ROI_%', text='ROI_%', title='Ganancia Promedio por Tipo',
            color_continuous_scale='Greens'
        )
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

    with col_rend2:
        fig_dona = px.pie(
            df_rendimiento, values='Monto_Total', names='tipo_inversion',
            hole=0.4, title='Distribución de Inversión por Tipo'
        )
        st.plotly_chart(fig_dona, use_container_width=True)
    
    fig_combo = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monto Total', 'Ganancia Total', 'ROI %', 'Cantidad')
    )
    fig_combo.add_trace(go.Bar(x=df_rendimiento['tipo_inversion'], y=df_rendimiento['Monto_Total'], name='Monto', marker_color='blue'), row=1, col=1)
    fig_combo.add_trace(go.Bar(x=df_rendimiento['tipo_inversion'], y=df_rendimiento['Ganancia_Total'], name='Ganancia', marker_color='green'), row=1, col=2)
    fig_combo.add_trace(go.Bar(x=df_rendimiento['tipo_inversion'], y=df_rendimiento['ROI_%'], name='ROI', marker_color='orange'), row=2, col=1)
    fig_combo.add_trace(go.Bar(x=df_rendimiento['tipo_inversion'], y=df_rendimiento['Cantidad'], name='Cantidad', marker_color='purple'), row=2, col=2)
    fig_combo.update_layout(height=600, showlegend=False, title_text="Dashboard de Rendimiento")
    st.plotly_chart(fig_combo, use_container_width=True)
    
    st.markdown("### 🎯 Rendimiento por Cluster")
    df_cluster_rend = df_auditado.groupby('cluster').agg({
        'monto_inicial': ['sum', 'mean', 'count'],
        'ganancia_perdida_simulada': ['sum', 'mean'],
        'tasa_anual_simulada': 'mean'
    }).round(2)
    df_cluster_rend.columns = ['Monto_Total', 'Monto_Promedio', 'Cantidad', 'Ganancia_Total', 'Ganancia_Promedio', 'Tasa_Promedio']
    df_cluster_rend['ROI_%'] = (df_cluster_rend['Ganancia_Total'] / df_cluster_rend['Monto_Total'] * 100).round(2)
    df_cluster_rend = df_cluster_rend.reset_index()
    
    fig_cluster = px.bar(
        df_cluster_rend, x='cluster', y=['Monto_Total', 'Ganancia_Total'],
        title='Monto vs Ganancia por Cluster', barmode='group'
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.dataframe(df_rendimiento, use_container_width=True)

# ---------------------------------------------------------------
# TAB 4: DISTRIBUCIONES
# ---------------------------------------------------------------
with tab4:
    if show_distributions:
        st.subheader("📊 Análisis de Distribuciones")
        
        dist_vars = st.multiselect(
            "Selecciona variables",
            ['monto_inicial', 'valor_actual_simulado', 'ganancia_perdida_simulada', 'tasa_anual_simulada', 'dias_plazo'],
            default=['monto_inicial', 'ganancia_perdida_simulada', 'tasa_anual_simulada']
        )
        
        if dist_vars:
            n_vars = len(dist_vars)
            cols = min(2, n_vars)
            rows = (n_vars + cols - 1) // cols
            
            fig_dist = make_subplots(rows=rows, cols=cols, subplot_titles=[v.replace('_', ' ').title() for v in dist_vars])
            
            for idx, var in enumerate(dist_vars):
                row = (idx // cols) + 1
                col = (idx % cols) + 1
                fig_dist.add_trace(go.Histogram(x=df_auditado[var], name=var, nbinsx=30), row=row, col=col)
            
            fig_dist.update_layout(height=300 * rows, showlegend=False, title_text="Distribuciones")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        st.markdown("### 📦 Box Plots por Cluster")
        box_var = st.selectbox("Variable", ['monto_inicial', 'ganancia_perdida_simulada', 'tasa_anual_simulada'], index=1)
        fig_box = px.box(df_auditado, x='cluster', y=box_var, color='cluster', points='all', title=f'Distribución de {box_var}')
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("### 🎻 Violin Plots")
        fig_violin = px.violin(df_auditado, x='cluster', y='ganancia_perdida_simulada', color='cluster', box=True, points='all')
        st.plotly_chart(fig_violin, use_container_width=True)
        
        st.markdown("### ⚠️ Análisis de Riesgo")
        df_riesgo = df_auditado.copy()
        df_riesgo['riesgo_categoria'] = pd.cut(
            df_riesgo['tasa_anual_simulada'],
            bins=[-np.inf, 0.05, 0.10, 0.20, np.inf],
            labels=['Bajo', 'Medio', 'Alto', 'Muy Alto']
        )
        df_riesgo = df_riesgo[df_riesgo['riesgo_categoria'].notna()]
        
        if not df_riesgo.empty:
            col_risk1, col_risk2 = st.columns(2)
            with col_risk1:
                fig_riesgo = px.pie(
                    df_riesgo, names='riesgo_categoria', values='monto_inicial',
                    title='Distribución de Riesgo por Monto',
                    color='riesgo_categoria',
                    color_discrete_map={'Bajo': 'green', 'Medio': 'yellow', 'Alto': 'orange', 'Muy Alto': 'red'}
                )
                st.plotly_chart(fig_riesgo, use_container_width=True)
            
            with col_risk2:
                riesgo_cluster = df_riesgo.groupby(['riesgo_categoria', 'cluster'])['monto_inicial'].sum().reset_index()
                fig_riesgo_bar = px.bar(
                    riesgo_cluster, x='riesgo_categoria', y='monto_inicial', color='cluster',
                    title='Riesgo por Cluster', barmode='group'
                )
                st.plotly_chart(fig_riesgo_bar, use_container_width=True)
            
            riesgo_pivot = df_riesgo.pivot_table(values='monto_inicial', index='riesgo_categoria', columns='cluster', aggfunc='sum', fill_value=0)
            fig_heat_riesgo = px.imshow(
                riesgo_pivot, text_auto='.0f', aspect="auto",
                color_continuous_scale='YlOrRd', title='Heatmap: Riesgo vs Cluster'
            )
            st.plotly_chart(fig_heat_riesgo, use_container_width=True)

# ---------------------------------------------------------------
# TAB 5: TEMPORAL
# ---------------------------------------------------------------
with tab5:
    if show_time_series and 'fecha_inicio' in df_auditado.columns:
        st.subheader("⏱️ Análisis Temporal")
        
        df_temp = df_auditado[df_auditado['fecha_inicio'].notna()].copy()
        
        if not df_temp.empty:
            df_temp['mes'] = df_temp['fecha_inicio'].dt.to_period('M').astype(str)
            df_temporal = df_temp.groupby('mes').agg({
                'monto_inicial': 'sum',
                'ganancia_perdida_simulada': 'sum',
                'inversion_id': 'count'
            }).reset_index()
            df_temporal.columns = ['Mes', 'Monto_Total', 'Ganancia_Total', 'Cantidad']
            df_temporal = df_temporal.sort_values('Mes')
            
            fig_time = make_subplots(rows=2, cols=1, subplot_titles=('Evolución del Monto', 'Evolución de Ganancia'))
            fig_time.add_trace(go.Scatter(x=df_temporal['Mes'], y=df_temporal['Monto_Total'], mode='lines+markers', name='Monto'), row=1, col=1)
            fig_time.add_trace(go.Scatter(x=df_temporal['Mes'], y=df_temporal['Ganancia_Total'], mode='lines+markers', name='Ganancia'), row=2, col=1)
            fig_time.update_layout(height=600, showlegend=False, title_text="Series Temporales")
            st.plotly_chart(fig_time, use_container_width=True)
            
            fig_time_bar = px.bar(df_temporal, x='Mes', y=['Monto_Total', 'Ganancia_Total'], title='Comparación Mensual', barmode='group')
            st.plotly_chart(fig_time_bar, use_container_width=True)
            
            df_temp['año'] = df_temp['fecha_inicio'].dt.year
            df_anual = df_temp.groupby('año').agg({'monto_inicial': 'sum', 'ganancia_perdida_simulada': 'sum'}).reset_index()
            
            fig_anual = px.bar(df_anual, x='año', y=['monto_inicial', 'ganancia_perdida_simulada'], title='Resumen Anual', barmode='group')
            st.plotly_chart(fig_anual, use_container_width=True)
        else:
            st.info("No hay datos temporales disponibles.")

# ---------------------------------------------------------------
# TAB 6: ANÁLISIS AVANZADO
# ---------------------------------------------------------------
with tab6:
    st.subheader("🔬 Análisis Avanzado")
    
    st.markdown("### 📊 Estadísticas Descriptivas Completas")
    st.dataframe(df_auditado.describe().round(2), use_container_width=True)
    
    st.markdown("### 🔥 Heatmap de Métricas por Tipo y Estado")
    pivot_heat = df_auditado.pivot_table(
        values='monto_inicial', index='tipo_inversion',
        columns='estado_inversion', aggfunc='sum', fill_value=0
    )
    fig_heat = px.imshow(pivot_heat, text_auto='.0f', aspect="auto", color_continuous_scale='YlOrRd', title='Monto por Tipo y Estado')
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("### 📈 Scatter Matrix")
    fig_scatter_matrix = px.scatter_matrix(
        df_auditado,
        dimensions=['monto_inicial', 'valor_actual_simulado', 'ganancia_perdida_simulada', 'tasa_anual_simulada'],
        color='cluster', title='Matriz de Dispersión'
    )
    fig_scatter_matrix.update_layout(height=700)
    st.plotly_chart(fig_scatter_matrix, use_container_width=True)
    
    st.markdown("### 🎯 Parallel Coordinates")
    fig_parallel = px.parallel_coordinates(
        df_auditado,
        dimensions=['monto_inicial', 'valor_actual_simulado', 'ganancia_perdida_simulada', 'tasa_anual_simulada'],
        color='cluster', color_continuous_scale=px.colors.diverging.Tealrose,
        title='Coordenadas Paralelas por Cluster'
    )
    st.plotly_chart(fig_parallel, use_container_width=True)
    
    st.markdown("### 📊 Radar Chart por Cluster")
    radar_data = df_auditado.groupby('cluster')[['monto_inicial', 'valor_actual_simulado', 'ganancia_perdida_simulada', 'tasa_anual_simulada']].mean()
    radar_data_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
    
    fig_radar = go.Figure()
    for cluster in radar_data_norm.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data_norm.loc[cluster].values.tolist() + [radar_data_norm.loc[cluster].values[0]],
            theta=list(radar_data_norm.columns) + [radar_data_norm.columns[0]],
            fill='toself', name=f'Cluster {cluster}'
        ))
    fig_radar.update_layout(title='Perfil de Clusters', polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------------------
# TAB 7: DATOS
# ---------------------------------------------------------------
with tab7:
    st.subheader("📋 Datos Completos")
    
    col_filt1, col_filt2, col_filt3 = st.columns(3)
    with col_filt1:
        filtro_cluster = st.multiselect("Cluster", sorted(df_auditado['cluster'].unique()), default=sorted(df_auditado['cluster'].unique()))
    with col_filt2:
        filtro_estado = st.multiselect("Estado", df_auditado['estado_inversion'].unique(), default=list(df_auditado['estado_inversion'].unique()))
    with col_filt3:
        filtro_tipo = st.multiselect("Tipo", df_auditado['tipo_inversion'].unique(), default=list(df_auditado['tipo_inversion'].unique()))
    
    df_filtrado = df_auditado[
        (df_auditado['cluster'].isin(filtro_cluster)) &
        (df_auditado['estado_inversion'].isin(filtro_estado)) &
        (df_auditado['tipo_inversion'].isin(filtro_tipo))
    ]
    
    st.markdown(f"**Registros:** {len(df_filtrado)} de {len(df_auditado)}")
    st.dataframe(df_filtrado, use_container_width=True)
    
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV", data=csv, file_name='inversiones_filtrado.csv', mime='text/csv')
