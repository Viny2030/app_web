import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Auditoría de Conciliación", layout="wide", page_icon="📊")

# --- TÍTULO Y ESTILO ---
st.title("🔍 Sistema de Auditoría y Conciliación Bancaria")
st.markdown("""
Esta aplicación utiliza **algoritmos de Machine Learning** para detectar anomalías en la conciliación
y agrupar discrepancias financieras.
""")


# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    # Cargamos el archivo CSV
    try:
        df = pd.read_csv('informe_conciliacion.csv')
        # Convertimos fechas
        df['fecha_banco'] = pd.to_datetime(df['fecha_banco'], errors='coerce')
        df['fecha_contable'] = pd.to_datetime(df['fecha_contable'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("Error: No se encuentra el archivo 'informe_conciliacion.csv'.")
        return pd.DataFrame()


df = load_data()

if not df.empty:
    # --- BARRA LATERAL ---
    st.sidebar.header("Filtros")
    estado_filtro = st.sidebar.multiselect(
        "Filtrar por Estado:",
        options=df['clasificación_auditoría'].unique(),
        default=df['clasificación_auditoría'].unique()
    )

    df_filtered = df[df['clasificación_auditoría'].isin(estado_filtro)]

    # --- KPIS PRINCIPALES ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transacciones", len(df_filtered))

    diferencia_total = df_filtered['diferencia_monto'].sum()
    col2.metric("Diferencia Total ($)", f"${diferencia_total:,.2f}", delta_color="inverse")

    no_conciliados = df_filtered[df_filtered['conciliado'] == False].shape[0]
    col3.metric("Pendientes de Revisión", no_conciliados)

    promedio_dias = df_filtered['diferencia_dias'].mean()
    col4.metric("Promedio Desfase (Días)", f"{promedio_dias:.1f} días")

    st.markdown("---")

    # --- PESTAÑAS DE ANÁLISIS ---
    tab1, tab2, tab3 = st.tabs(["📊 Resumen Visual", "🤖 Detección de Fraude (IA)", "📋 Datos Detallados"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribución de Clasificaciones")
            fig_pie = px.pie(df_filtered, names='clasificación_auditoría', title='Estado de la Conciliación')
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("Diferencias Monetarias")
            fig_hist = px.histogram(df_filtered, x='diferencia_monto', title='Histograma de Diferencias ($)')
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.subheader("Algoritmos de Detección de Anomalías")
        st.info(
            "El algoritmo **Isolation Forest** analiza las diferencias de monto y días para aislar casos sospechosos.")

        # PREPARACIÓN PARA ML
        # Seleccionamos solo columnas numéricas relevantes para buscar anomalías
        features = ['diferencia_monto', 'diferencia_dias']
        X = df_filtered[features].fillna(0)

        if len(X) > 0:
            # ALGORITMO 1: ISOLATION FOREST
            model_iso = IsolationForest(contamination=0.05, random_state=42)
            df_filtered['anomalia_score'] = model_iso.fit_predict(X)
            # -1 es anomalía, 1 es normal
            anomalias = df_filtered[df_filtered['anomalia_score'] == -1]

            st.error(
                f"⚠️ Se han detectado **{len(anomalias)} transacciones anómalas** que requieren atención inmediata.")

            # Visualización de Anomalías
            fig_scatter = px.scatter(
                df_filtered,
                x='diferencia_dias',
                y='diferencia_monto',
                color=df_filtered['anomalia_score'].map({1: 'Normal', -1: 'Anomalía'}),
                color_discrete_map={'Normal': 'blue', 'Anomalía': 'red'},
                hover_data=['concepto_banco', 'fecha_banco'],
                title="Mapa de Calor de Anomalías (Monto vs Días)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.write("#### 🚨 Listado de Anomalías Detectadas")
            st.dataframe(
                anomalias[['fecha_banco', 'concepto_banco', 'monto_banco', 'diferencia_monto', 'diferencia_dias']])

    with tab3:
        st.subheader("Explorador de Datos")
        st.dataframe(
            df_filtered.style.applymap(lambda x: 'color: red' if x == False else 'color: green', subset=['conciliado']))

else:
    st.warning("Esperando archivo de datos...")