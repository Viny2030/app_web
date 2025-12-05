TEMPLATE = """
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Reporte Automático", layout="wide")

# Cargar datos
try:
    df = pd.read_csv("dataset.csv")
    
    st.title("📊 Reporte Automático del Dataset")
    st.markdown("### Tipo de Problema Detectado: **{tipo_problema}**")

    # Información del dataset
    st.subheader("Información del Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Número de filas", len(df))
    with col2:
        st.metric("Número de columnas", len(df.columns))

    # Vista previa del dataset
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(), use_container_width=True)

    # Resultados del modelo
    st.subheader("Resultados del Modelo")
    try:
        resultados = json.loads('{resultados_json}')
        st.json(resultados)
    except Exception as e:
        st.error(f"Error al cargar los resultados: {str(e)}")
        st.code('{resultados_json}')

    # ----------------------------------------------------------------------
    # 💡 SECCIÓN DE GRÁFICOS
    # ----------------------------------------------------------------------
    
    tipo_problema = "{tipo_problema}"
    cluster_col = 'Cluster_ID' if 'Cluster_ID' in df.columns else 'Prediccion_ML'

if tipo_problema in ["clasificacion", "regresion"]:
    st.subheader("Predicciones vs Reales (Problema Supervisado)")
    try:
        # Se asume que Target_Real y Prediccion_ML existen en dataset.csv
        fig, ax = plt.subplots()
        ax.plot(df["Target_Real"], label="Real") 
        ax.plot(df["Prediccion_ML"], label="Predicho")
        ax.legend()
        st.pyplot(fig)

    except Exception:
        st.info("No se pudieron generar gráficos simples (Matplotlib) para este tipo de modelo supervisado.")

elif tipo_problema == "clustering_o_anomalias":
    st.subheader("Visualización de Clústeres (Plotly Interactivo)")

    try:
        # 1. Identificar columnas numéricas
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # 2. Eliminar columnas de ID y la columna Cluster_ID de las features si son numéricas
        numeric_features = [col for col in numeric_cols if col.lower() not in ['id', 'inversion_id', cluster_col.lower()]]

        if cluster_col in df.columns:
            # Convertir la columna Cluster_ID a string para usarla como color/símbolo en Plotly
            df[cluster_col] = df[cluster_col].astype(str)

            # 3. Generar el gráfico 3D si hay suficientes columnas numéricas
            if len(numeric_features) >= 3:
                fig = px.scatter_3d(
                    df, 
                    x=numeric_features[0],
                    y=numeric_features[1],
                    z=numeric_features[2],
                    color=cluster_col,
                    symbol=cluster_col,
                    title=f"Visualización 3D de {len(df[cluster_col].unique())} Clústeres",
                    hover_data=df.columns.tolist() # Mostrar todos los datos al pasar el ratón
                )
                st.plotly_chart(fig, use_container_width=True)

            # 4. Generar el gráfico 2D si hay solo 2 columnas numéricas
            elif len(numeric_features) >= 2:
                fig = px.scatter(
                    df,
                    x=numeric_features[0],
                    y=numeric_features[1],
                    color=cluster_col,
                    symbol=cluster_col,
                    title=f"Visualización 2D de {len(df[cluster_col].unique())} Clústeres",
                    hover_data=df.columns.tolist()
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No hay suficientes columnas numéricas (se necesitan al menos 2) para generar un gráfico de clústeres.")

        else:
            st.warning("No se encontró la columna de etiquetas de clústeres ('Cluster_ID') en los datos para graficar. Asegúrese de que el proceso ML guardó las predicciones.")

    except Exception as e:
        st.error(f"Error al generar el gráfico interactivo: {e}")

else:
    st.info("No se requiere una visualización estándar para este tipo de problema.")
"""