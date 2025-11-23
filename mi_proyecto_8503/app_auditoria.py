# =================================================================
# SCRIPT DE INTERFAZ STREAMLIT PARA AUDITORÍA DE PRODUCTOS EN PROCESO
# Para ejecutar: streamlit run app_auditoria.py
# =================================================================

# --- 1. IMPORTACIONES ---

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Importar las funciones de generación de datos y auditoría
from datos_auditoria import generar_datos_simulados, aplicar_auditoria


# ===============================================================
# 2. CONFIGURACIÓN DE PÁGINA Y CACHÉ
# ===============================================================

st.set_page_config(page_title="Auditoría de Productos en Proceso", layout="wide")

# En Streamlit, la función de generación y auditoría deben ser decoradas 
# para aprovechar el almacenamiento en caché y evitar regenerar los datos 
# en cada interacción.

@st.cache_data
def get_audit_data():
    """Genera y audita los datos, usando la caché de Streamlit."""
    df_proceso = generar_datos_simulados()
    df_auditado = aplicar_auditoria(df_proceso)
    return df_auditado


# ===============================================================
# 3. INTERFAZ DE STREAMLIT
# ===============================================================

st.title("⚙️ Auditoría de Productos en Proceso (WIP)")
st.markdown("Esta aplicación audita datos simulados de productos en proceso, identificando anomalías y aplicando reglas de negocio.")

if st.button("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner('Ejecutando la auditoría...'):
        df_auditado = get_audit_data()

        st.success("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header("🔍 Resultados de la Auditoría")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Procesos", len(df_auditado))
        with col2:
            anomalias_count = len(df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo'])
            st.metric("Anomalías Detectadas", anomalias_count)

        anomalies_and_alerts_df = df_auditado[
            (df_auditado['resultado_auditoria'] == 'Anómalo') | (df_auditado['alerta_heuristica'] != "Sin alertas")]

        st.subheader("Procesos Anómalos o con Alertas")
        if not anomalies_and_alerts_df.empty:
            columnas_interes = ['id_proceso', 'producto', 'etapa_actual', 'linea_produccion', 'cantidad_en_proceso',
                                'avance_porcentaje', 'estado', 'alerta_heuristica', 'resultado_auditoria']
            st.dataframe(anomalies_and_alerts_df[columnas_interes])

            csv_data = anomalies_and_alerts_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias_wip.csv",
                mime="text/csv"
            )
        else:
            st.info("¡No se encontraron anomalías o alertas significativas!")

        # --- Sección 2: Visualizaciones ---
        st.header("📈 Visualizaciones Clave")

        # Gráfico 1: Avance por Etapa
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        sns.boxplot(data=df_auditado, x='avance_porcentaje', y='etapa_actual', hue='resultado_auditoria',
                     palette={'Normal': 'skyblue', 'Anómalo': 'salmon'}, ax=ax1)
        ax1.set_title('Avance por Etapa de Producción')
        ax1.set_xlabel('Avance (%)')
        ax1.set_ylabel('Etapa Actual')
        st.pyplot(fig1)

        # Gráfico 2: Cantidad en Proceso vs Avance
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_auditado, x='cantidad_en_proceso', y='avance_porcentaje', hue='resultado_auditoria',
                         style='linea_produccion', palette={'Normal': 'green', 'Anómalo': 'red'}, alpha=0.8, s=100,
                         ax=ax2)
        ax2.set_title('Cantidad en Proceso vs Avance (%)')
        ax2.set_xlabel('Cantidad en Proceso (unidades)')
        ax2.set_ylabel('Avance (%)')
        st.pyplot(fig2)