
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Reporte Automático", layout="wide")

st.title("📊 Reporte Automático del Dataset")
st.markdown("### Tipo de Problema Detectado: **clustering_o_anomalias**")

# Dataset preview
st.subheader("Vista previa del dataset")
df = pd.read_csv("dataset.csv")
st.dataframe(df.head(), use_container_width=True)

# Resultados del modelo
st.subheader("Resultados del Modelo")
st.json({"tipo_problema": "clustering_o_anomalias", "clusters_encontrados": 3})

# Gráfico simple
st.subheader("Predicciones vs Reales")

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(real, label="Real")
    ax.plot(pred, label="Predicho")
    ax.legend()

    st.pyplot(fig)
except:
    st.info("No se pudieron generar gráficos para este tipo de modelo.")
