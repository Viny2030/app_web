# =================================================================
# auditoria_prestamos.py (Lee desde Backblaze B2 y Audita)
# =================================================================

# --- 1. IMPORTACIONES UNIFICADAS ---
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import streamlit as st

# Importar la función de DESCARGA de datos desde el módulo externo
from generador_datos import obtener_dataset_prestamos_de_b2

# =================================================================
# 2. CONFIGURACIÓN DE PÁGINA
# =================================================================

st.set_page_config(page_title="Auditoría de Préstamos Obtenidos", layout="wide")


# =================================================================
# 3. LÓGICA DE AUDITORÍA
# =================================================================

def aplicar_auditoria(df):
    """Aplica las reglas heurísticas y el modelo de detección de anomalías."""
    df['Fecha_Obtencion'] = pd.to_datetime(df['Fecha_Obtencion'])
    numeric_cols = ['Monto_Prestamo', 'Tasa_Interes_Anual', 'Plazo_Meses']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True)

    # Variables derivadas
    df['Pago_Total_Proyectado'] = df['Monto_Prestamo'] * (1 + df['Tasa_Interes_Anual'] * (df['Plazo_Meses'] / 12))
    fecha_actual_referencia = datetime.now()
    df['Dias_Desde_Obtencion'] = (fecha_actual_referencia - df['Fecha_Obtencion']).dt.days

    # Detección de anomalías basada en Z-Score (heurística simple)
    df['monto_zscore'] = zscore(df['Monto_Prestamo'])
    df['tasa_zscore'] = zscore(df['Tasa_Interes_Anual'])

    # Detección de anomalías con Isolation Forest (IA)
    features_for_anomaly_detection = df[['Monto_Prestamo', 'Tasa_Interes_Anual', 'Plazo_Meses']].copy()
    features_for_anomaly_detection.fillna(features_for_anomaly_detection.median(), inplace=True)
    iso_forest = IsolationForest(random_state=42, contamination=0.1)
    iso_forest.fit(features_for_anomaly_detection)
    df['is_anomaly_ia'] = iso_forest.predict(features_for_anomaly_detection)  # -1 para anomalía, 1 para normal

    return df


# =================================================================
# 4. INTERFAZ DE STREAMLIT
# =================================================================

st.title("💸 Auditoría de Préstamos Obtenidos")
st.markdown(
    "Esta aplicación audita datos de préstamos, obtenidos desde **Backblaze B2**, identificando anomalías en montos y tasas de interés.")

if st.button("Iniciar Auditoría", help="Descarga el dataset de Backblaze B2 y aplica el análisis completo"):
    with st.spinner('Descargando datos desde Backblaze B2 y ejecutando la auditoría...'):
        # Llama a la función de descarga de B2
        df_prestamos = obtener_dataset_prestamos_de_b2(bucket_name="dataset-raw", file_key="prestamos_simulados.csv")

        if df_prestamos.empty:
            st.warning("No se pudo iniciar la auditoría. Revisa el informe de error en la configuración de B2.")
        else:
            df_auditado = aplicar_auditoria(df_prestamos)

            st.success(f"✅ Auditoría completada con éxito. Procesados {len(df_auditado)} registros.")

            # --- Sección 1: Resumen y Alertas ---
            st.header("🔍 Informe de Auditoría")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Préstamos", len(df_auditado))
            with col2:
                monto_total = df_auditado['Monto_Prestamo'].sum()
                st.metric("Monto Total de Préstamos", f"${monto_total:,.2f}")
            with col3:
                anomalias_ia_count = (df_auditado['is_anomaly_ia'] == -1).sum()
                st.metric("Anomalías por IA", anomalias_ia_count)

            st.subheader("Resumen de Estados de Pago")
            st.dataframe(df_auditado['Estado_Pago'].value_counts())

            anomalias_ia_df = df_auditado[df_auditado['is_anomaly_ia'] == -1]

            if not anomalias_ia_df.empty:
                st.subheader("Préstamos con Anomalías Detectadas")
                st.dataframe(anomalias_ia_df[['ID_Prestamo', 'Monto_Prestamo', 'Tasa_Interes_Anual', 'Plazo_Meses']])
                csv_data = anomalias_ia_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar Reporte de Anomalías CSV",
                    data=csv_data,
                    file_name="reporte_anomalias_prestamos.csv",
                    mime="text/csv"
                )
            else:
                st.info("No se detectaron anomalías por Isolation Forest.")

            # --- Sección 2: Visualizaciones ---
            st.header("📈 Visualizaciones")

            # Gráficos de distribución
            col_viz1, col_viz2, col_viz3 = st.columns(3)
            with col_viz1:
                fig1, ax1 = plt.subplots()
                sns.histplot(df_auditado['Monto_Prestamo'], bins=10, kde=True, color='skyblue', ax=ax1)
                ax1.set_title('1. Distribución del Monto de Préstamo')
                st.pyplot(fig1)

            with col_viz2:
                fig2, ax2 = plt.subplots()
                sns.histplot(df_auditado['Tasa_Interes_Anual'], bins=10, kde=True, color='lightgreen', ax=ax2)
                ax2.set_title('2. Distribución de la Tasa de Interés')
                st.pyplot(fig2)

            with col_viz3:
                fig3, ax3 = plt.subplots()
                sns.histplot(df_auditado['Plazo_Meses'], bins=5, kde=True, color='salmon', ax=ax3)
                ax3.set_title('3. Distribución del Plazo (Meses)')
                st.pyplot(fig3)

            # Gráfico 4: Distribución por Estado
            fig4, ax4 = plt.subplots()
            df_auditado['Estado_Pago'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90,
                                                           colors=sns.color_palette("pastel"), ax=ax4)
            ax4.set_title('4. Distribución de Préstamos por Estado de Pago')
            ax4.set_ylabel('')
            st.pyplot(fig4)

            # Gráfico 5: Detección de Anomalías con IA
            st.subheader("Detección de Anomalías por Isolation Forest")
            fig5, ax5 = plt.subplots(figsize=(12, 8))
            sns.scatterplot(
                x='Monto_Prestamo',
                y='Tasa_Interes_Anual',
                hue='is_anomaly_ia',
                data=df_auditado,
                palette={1: 'blue', -1: 'red'},
                style='is_anomaly_ia',
                markers={1: 'o', -1: 'X'},
                s=100,
                ax=ax5
            )
            ax5.set_title('Monto vs. Tasa de Interés')
            ax5.set_xlabel('Monto del Préstamo')
            ax5.set_ylabel('Tasa de Interés Anual')
            handles, labels = ax5.get_legend_handles_labels()
            ax5.legend(handles, ['Normal', 'Anomalía'], title='Resultado IA')
            st.pyplot(fig5)