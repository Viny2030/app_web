import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Análisis de Datos Low-Code", layout="wide")

# Título de la aplicación
st.title("📊 Analizador de Datos Low-Code")
st.write("Carga tus datos y explóralos de forma interactiva")

# 1. Carga de datos
st.sidebar.header("1. Cargar Datos")
archivo = st.sidebar.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])

if archivo is not None:
    # Leer el archivo
    if archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)
    
    # Mostrar vista previa de los datos
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())
    
    # Estadísticas básicas
    st.subheader("📈 Estadísticas básicas")
    st.write(df.describe())
    
    # 2. Selección de gráfico
    st.sidebar.header("2. Crear Gráfico")
    tipo_grafico = st.sidebar.selectbox(
        "Tipo de gráfico",
        ["Histograma", "Dispersión", "Barras", "Líneas"]
    )
    
    # Obtener columnas numéricas
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if tipo_grafico == "Histograma" and columnas_numericas:
        col = st.sidebar.selectbox("Selecciona una columna", columnas_numericas)
        fig = px.histogram(df, x=col, title=f"Distribución de {col}")
        st.plotly_chart(fig, use_container_width=True)
        
    elif tipo_grafico == "Dispersión" and len(columnas_numericas) >= 2:
        col_x = st.sidebar.selectbox("Eje X", columnas_numericas)
        col_y = st.sidebar.selectbox("Eje Y", [c for c in columnas_numericas if c != col_x])
        fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_y} vs {col_x}")
        st.plotly_chart(fig, use_container_width=True)
        
    elif tipo_grafico == "Barras" and columnas_numericas:
        col_x = st.sidebar.selectbox("Categoría", df.columns)
        col_y = st.sidebar.selectbox("Valor", [c for c in columnas_numericas if c != col_x])
        fig = px.bar(df, x=col_x, y=col_y, title=f"{col_y} por {col_x}")
        st.plotly_chart(fig, use_container_width=True)
        
    elif tipo_grafico == "Líneas" and len(columnas_numericas) >= 1:
        col_x = st.sidebar.selectbox("Eje X (fecha o categoría)", df.columns)
        col_y = st.sidebar.selectbox("Eje Y (valor)", columnas_numericas)
        fig = px.line(df, x=col_x, y=col_y, title=f"Evolución de {col_y}")
        st.plotly_chart(fig, use_container_width=True)
        
    # 3. Análisis rápido
    st.sidebar.header("3. Análisis Rápido")
    if st.sidebar.button("🔍 Mostrar información del dataset"):
        st.subheader("Información del Dataset")
        st.write(f"- Número de filas: {len(df)}")
        st.write(f"- Número de columnas: {len(df.columns)}")
        st.write("\n**Tipos de datos:**")
        st.write(df.dtypes.astype(str))
        
    # 4. Exportar resultados
    st.sidebar.header("4. Exportar")
    if st.sidebar.button("💾 Exportar datos procesados"):
        output = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Descargar CSV",
            data=output,
            file_name="datos_procesados.csv",
            mime="text/csv"
        )
else:
    st.info("👈 Por favor, sube un archivo CSV o Excel para comenzar")
    
# Instrucciones
with st.expander("ℹ️ Cómo usar esta aplicación"):
    st.markdown("""
    1. **Sube tus datos**: Usa el panel de la izquierda para subir un archivo CSV o Excel
    2. **Explora los datos**: Visualiza una vista previa y estadísticas básicas
    3. **Crea gráficos**: Selecciona el tipo de gráfico que deseas generar
    4. **Exporta resultados**: Descarga los datos procesados si lo necesitas
    
    No se requiere programación - ¡todo se controla mediante la interfaz!
    """)
