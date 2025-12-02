TEMPLATE = """
import streamlit as st
import pandas as pd
import json

# ==============================================================================
# 🎯 CONFIGURACIÓN
# ==============================================================================
st.set_page_config(page_title="Reporte Automático", layout="wide")

# ==============================================================================
# 💡 VARIABLES PASADAS POR generar_dashboard.py
# ==============================================================================
# Estos son los datos inyectados por la función generar_dashboard.py:
# pred = {pred}
# real = {real}
# resultados_dict = json.loads('{resultados_json}')
# tipo_problema = "{tipo_problema}"

# Cargamos el diccionario de resultados para obtener el tipo de problema y scores
resultados_dict = json.loads('{resultados_json}')
tipo = resultados_dict['tipo_problema']

# ==============================================================================
# 📊 ESTRUCTURA DEL DASHBOARD
# ==============================================================================

st.title("📊 Reporte Automático del Dataset")
st.markdown(f"### Tipo de Problema Detectado: **{{tipo_problema}}**")

# --- Vista previa del dataset ---
st.subheader("Vista previa del dataset")
# Asumimos que dataset.csv contiene el dataframe original más la columna 'target' (si aplica)
try:
    df = pd.read_csv("dataset.csv")
    st.dataframe(df.head(), use_container_width=True)
except Exception as e:
    st.error(f"Error al cargar el dataset.csv: {{e}}")

# --- Resultados del modelo ---
st.subheader("Resultados del Modelo")
st.json(resultados_dict)

# --- Gráficos (Lógica Condicional) ---
st.subheader("Predicciones vs Reales")

# Convertimos las listas de Python inyectadas a objetos (aunque pueden ser strings)
pred = {pred}
real = {real}


if tipo == "clustering_o_anomalias":
    try:
        # Importamos Plotly para un gráfico de dispersión interactivo (ideal para clustering)
        import plotly.express as px

        # El clustering requiere que al menos haya dos columnas numéricas para graficar en 2D.
        # Seleccionamos solo las columnas numéricas del DataFrame
        df_num = df.select_dtypes(include=['number'])

        if df_num.shape[1] >= 2 and pred:

            # Agregamos las etiquetas de cluster (predicciones) como una nueva columna al DataFrame
            df['Cluster'] = pd.Series(pred).astype(str)

            # Graficar las dos primeras columnas numéricas coloreadas por el cluster
            fig = px.scatter(
                df, 
                x=df_num.columns[0], 
                y=df_num.columns[1], 
                color='Cluster', 
                title='Clustering de Puntos de Datos'
            )
            st.plotly_chart(fig, use_container_width=True)

        elif not pred:
             st.info("El modelo no generó etiquetas de cluster.")
        else:
            st.info("El dataset no tiene suficientes columnas numéricas para generar un gráfico de dispersión de clustering en 2D.")

    except Exception as e:
        st.error(f"❌ Error al generar el gráfico de dispersión (Plotly): {{e}}")

elif tipo in ["clasificacion", "regresion"]:
    try:
        # Importamos Matplotlib para el gráfico de línea (Predicciones vs Reales)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        # Aseguramos que la longitud de las listas sea la misma antes de graficar
        longitud = min(len(real), len(pred))

        ax.plot(real[:longitud], label="Real", color='blue')
        ax.plot(pred[:longitud], label="Predicho", color='red', linestyle='--')
        ax.set_title(f"Predicciones vs Reales ({tipo.capitalize()})")
        ax.legend()

        st.pyplot(fig)
    except Exception as e:
        st.info("No se pudieron generar gráficos (Fallo al procesar las series de datos).")

else:
    st.info("No se requiere o no está implementada una visualización estándar para este tipo de problema.")

"""