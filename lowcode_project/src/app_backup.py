import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, 
    mean_squared_error, r2_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Datos Low-Code",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título de la aplicación
st.title("📊 Análisis de Datos Low-Code")
st.write("Carga, explora y visualiza tus datos sin necesidad de programar")

# --- Variables globales ---
UPLOAD_FOLDER = Path("../data/uploaded")
MODEL_FOLDER = Path("../models")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

# Almacenamiento de sesión
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# --- Funciones de utilidad ---
def load_data(file):
    """Carga datos desde diferentes formatos de archivo."""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            st.error("Formato de archivo no soportado. Por favor, sube un archivo CSV o Excel.")
            return None
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

def preprocess_data(df, target_col):
    """Preprocesa los datos para el modelo."""
    df = df.copy()
    
    # Eliminar filas con valores faltantes en la columna objetivo
    df = df.dropna(subset=[target_col])
    
    # Codificar variables categóricas
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_col:
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Codificar la variable objetivo si es categórica
    if df[target_col].dtype == 'object':
        df[target_col] = le.fit_transform(df[target_col])
    
    # Separar características y objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Estandarizar características
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, model_type='clasificacion'):
    """Entrena un modelo de machine learning."""
    if model_type == 'clasificacion':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:  # regresión
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_type='clasificacion'):
    """Evalúa el modelo y devuelve métricas."""
    y_pred = model.predict(X_test)
    
    if model_type == 'clasificacion':
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    else:  # regresión
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            'mse': mse,
            'r2': r2,
            'predictions': y_pred
        }

def plot_feature_importance(model, feature_names):
    """Genera un gráfico de importancia de características."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[feature_names[i] for i in indices],
        y=importances[indices],
        marker_color='skyblue'
    ))
    
    fig.update_layout(
        title='Importancia de las características',
        xaxis_title='Características',
        yaxis_title='Importancia',
        template='plotly_white',
        height=500
    )
    
    return fig

# --- Barra lateral ---
st.sidebar.header("Configuración")

# Selector de tema
theme = st.sidebar.selectbox(
    "Tema de la aplicación",
    ["Claro", "Oscuro"],
    index=0
)

# --- Sección de carga de datos ---
st.sidebar.header("1. Cargar Datos")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo de datos",
    type=["csv", "xlsx", "xls"],
    help="Formatos soportados: CSV, Excel (XLSX, XLS)"
)

# Procesar archivo cargado
df = None
if uploaded_file is not None:
    # Guardar archivo
    file_path = UPLOAD_FOLDER / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Cargar datos
    df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.success(f"Archivo cargado: {uploaded_file.name}")
        st.sidebar.write(f"- Filas: {len(df)}")
        st.sidebar.write(f"- Columnas: {len(df.columns)}")

# --- Sección principal ---
if df is not None:
    # Pestañas para diferentes vistas
    tab1, tab2, tab3 = st.tabs(["📋 Datos", "📊 Análisis", "📈 Visualización"])
    
    with tab1:
        st.header("Vista previa de los datos")
        st.dataframe(df.head())
        
        # Mostrar información del dataset
        with st.expander("📋 Información del dataset"):
            st.write("**Tipos de datos:**")
            st.write(df.dtypes.astype(str))
            
            st.write("\n**Valores faltantes:**")
            missing = df.isnull().sum()
            st.bar_chart(missing[missing > 0])
    
    with tab2:
        st.header("Análisis predictivo")
        
        # Selección de modelo
        st.subheader("Configuración del modelo")
        
        # Seleccionar columna objetivo
        target_col = st.selectbox(
            "Selecciona la variable objetivo (target)",
            df.columns,
            key="target_col"
        )
        
        # Determinar tipo de problema
        problem_type = st.radio(
            "Tipo de problema",
            ["Clasificación", "Regresión"],
            key="problem_type"
        )
        
        if st.button("Entrenar modelo"):
            with st.spinner("Entrenando modelo..."):
                try:
                    # Preprocesar datos
                    X_train, X_test, y_train, y_test = preprocess_data(df, target_col)
                    
                    # Entrenar modelo
                    model = train_model(
                        X_train, 
                        y_train, 
                        model_type='clasificacion' if problem_type == 'Clasificación' else 'regresion'
                    )
                    
                    # Evaluar modelo
                    metrics = evaluate_model(
                        model, 
                        X_test, 
                        y_test,
                        model_type='clasificacion' if problem_type == 'Clasificación' else 'regresion'
                    )
                    
                    # Guardar en sesión
                    st.session_state.model = model
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.metrics = metrics
                    
                    st.success("¡Modelo entrenado exitosamente!")
                    
                except Exception as e:
                    st.error(f"Error al entrenar el modelo: {str(e)}")
        
        # Mostrar resultados si el modelo está entrenado
        if st.session_state.model is not None:
            st.subheader("Resultados del modelo")
            
            if problem_type == 'Clasificación':
                # Métricas de clasificación
                st.metric("Precisión", f"{st.session_state.metrics['accuracy']:.2f}")
                
                # Matriz de confusión
                st.subheader("Matriz de confusión")
                fig = px.imshow(
                    st.session_state.metrics['confusion_matrix'],
                    text_auto=True,
                    labels=dict(x="Predicho", y="Real", color="Cantidad"),
                    x=[f'Clase {i}' for i in range(len(st.session_state.metrics['confusion_matrix']))],
                    y=[f'Clase {i}' for i in range(len(st.session_state.metrics['confusion_matrix']))]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Reporte de clasificación
                st.subheader("Reporte de clasificación")
                report_df = pd.DataFrame(st.session_state.metrics['report']).transpose()
                st.dataframe(report_df)
                
            else:  # Regresión
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Error Cuadrático Medio (MSE)", f"{st.session_state.metrics['mse']:.4f}")
                with col2:
                    st.metric("R² Score", f"{st.session_state.metrics['r2']:.4f}")
                
                # Gráfico de valores reales vs predichos
                st.subheader("Valores reales vs Predichos")
                fig = px.scatter(
                    x=st.session_state.y_test,
                    y=st.session_state.metrics['predictions'],
                    labels={'x': 'Valores Reales', 'y': 'Predicciones'},
                    trendline="ols"
                )
                fig.add_trace(go.Scatter(
                    x=[min(st.session_state.y_test), max(st.session_state.y_test)],
                    y=[min(st.session_state.y_test), max(st.session_state.y_test)],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Línea de referencia'
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            # Importancia de características
            st.subheader("Importancia de las características")
            importance_fig = plot_feature_importance(st.session_state.model, X_test.columns)
            st.plotly_chart(importance_fig, use_container_width=True)
            
            # Botón para guardar el modelo
            if st.button("💾 Guardar modelo"):
                model_path = MODEL_FOLDER / "modelo_entrenado.pkl"
                joblib.dump(st.session_state.model, model_path)
                st.success(f"Modelo guardado en {model_path}")
    
    with tab3:
        st.header("Visualización de datos")
        
        # Selector de tipo de gráfico
        chart_type = st.selectbox(
            "Tipo de gráfico",
            ["Histograma", "Dispersión", "Barras", "Líneas"]
        )
        
        # Configuración del gráfico según el tipo seleccionado
        if chart_type == "Histograma" and numeric_cols:
            col = st.selectbox("Selecciona una columna numérica", numeric_cols)
            fig = px.histogram(df, x=col, title=f"Distribución de {col}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Dispersión" and len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Eje X", numeric_cols)
            with col2:
                y_axis = st.selectbox("Eje Y", [c for c in numeric_cols if c != x_axis])
            
            color_col = st.selectbox(
                "Color por (opcional)",
                ["Ninguno"] + df.columns.tolist()
            )
            
            fig = px.scatter(
                df, 
                x=x_axis, 
                y=y_axis, 
                color=color_col if color_col != "Ninguno" else None,
                title=f"{y_axis} vs {x_axis}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Barras" and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Categoría", df.columns)
            with col2:
                y_axis = st.selectbox("Valor", numeric_cols)
                
            fig = px.bar(
                df, 
                x=x_axis, 
                y=y_axis, 
                title=f"{y_axis} por {x_axis}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Líneas" and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Eje X (fecha o categoría)", df.columns)
            with col2:
                y_axis = st.selectbox("Eje Y (valor)", numeric_cols)
                
            fig = px.line(
                df, 
                x=x_axis, 
                y=y_axis, 
                title=f"Evolución de {y_axis}"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    # Página de bienvenida cuando no hay datos cargados
    st.markdown("""
    ## 👋 ¡Bienvenido al Analizador de Datos Low-Code con Machine Learning!
    
    Esta aplicación te permite analizar, visualizar y modelar tus datos sin necesidad de escribir código.
    
    ### Cómo comenzar:
    1. Usa el panel de la izquierda para subir un archivo de datos (CSV o Excel)
    2. Explora tus datos en las diferentes pestañas
    3. Entrena modelos de Machine Learning
    4. Visualiza los resultados y métricas
    
    ### Características principales:
    - 📊 Visualización interactiva de datos
    - 🤖 Modelos de Machine Learning (Clasificación y Regresión)
    - 📊 Gráficos interactivos con Plotly
    - 📊 Análisis de importancia de características
    - 📝 Reportes detallados de los modelos
    
    ### Requisitos de datos:
    - Archivos CSV o Excel
    - Datos limpios (sin valores faltantes en la variable objetivo)
    - Para clasificación: la variable objetivo debe ser categórica
    - Para regresión: la variable objetivo debe ser numérica
    """)

# --- Pie de página ---
st.sidebar.markdown("---")
st.sidebar.info(
    """
    ### Acerca de
    Versión 1.0.0  
    Desarrollado con ❤️ usando Streamlit
    """
)
