import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import sys
import time
from datetime import datetime
import json

# Agregar el directorio raíz al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

# Importar servicios
from services.backblaze_service import BackblazeService
from services.analisis_service import AnalizadorDatos
from services.notificacion_service import ServicioNotificaciones
from config import (
    B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET_INPUT, B2_BUCKET_OUTPUT,
    UPLOAD_DIR, MODEL_DIR, EMAIL_FROM
)

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Datos Automatizado",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título de la aplicación
st.title("🤖 Análisis de Datos Automatizado")
st.write("Carga tus datos, analízalos automáticamente y obtén insights valiosos")

# Inicializar servicios
try:
    backblaze = BackblazeService(B2_KEY_ID, B2_APPLICATION_KEY)
    notificador = ServicioNotificaciones()
except Exception as e:
    st.error(f"Error al inicializar los servicios: {str(e)}")
    st.stop()

# Variables de sesión
if 'analizador' not in st.session_state:
    st.session_state.analizador = None
if 'analisis_completado' not in st.session_state:
    st.session_state.analisis_completado = False
if 'enlace_resultados' not in st.session_state:
    st.session_state.enlace_resultados = None

# --- Funciones de utilidad ---
def cargar_datos(archivo):
    """Carga datos desde diferentes formatos de archivo."""
    try:
        if archivo.name.endswith('.csv'):
            return pd.read_csv(archivo)
        elif archivo.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(archivo)
        else:
            st.error("Formato de archivo no soportado. Por favor, sube un archivo CSV o Excel.")
            return None
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

def guardar_resultados(analizador, nombre_archivo, email_cliente):
    """Guarda los resultados del análisis y devuelve un enlace de descarga."""
    try:
        # Crear directorio temporal para resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        carpeta_resultados = MODEL_DIR / f"resultados_{timestamp}"
        carpeta_resultados.mkdir(exist_ok=True)
        
        # Guardar modelo
        ruta_modelo = carpeta_resultados / "modelo.joblib"
        analizador.guardar_modelo(str(ruta_modelo))
        
        # Crear informe de resultados
        informe = {
            'fecha_analisis': datetime.now().isoformat(),
            'nombre_archivo': nombre_archivo,
            'email_cliente': email_cliente,
            'tipo_problema': analizador.tipo_problema,
            'columna_objetivo': analizador.columna_objetivo,
            'metricas': {}
        }
        
        # Guardar informe
        with open(carpeta_resultados / 'informe.json', 'w') as f:
            json.dump(informe, f, indent=2)
        
        # Comprimir resultados
        import shutil
        shutil.make_archive(str(carpeta_resultados), 'zip', carpeta_resultados)
        
        # Subir a Backblaze
        ruta_zip = f"{carpeta_resultados}.zip"
        enlace = backblaze.upload_file(ruta_zip, B2_BUCKET_OUTPUT, f"resultados/{Path(ruta_zip).name}")
        
        return enlace
        
    except Exception as e:
        st.error(f"Error al guardar resultados: {e}")
        return None

def mostrar_metricas(metricas, tipo_problema):
    """Muestra las métricas del modelo de forma interactiva."""
    if tipo_problema == 'clasificacion':
        st.metric("Exactitud", f"{metricas['exactitud']:.2%}")
        
        st.subheader("Reporte de Clasificación")
        st.json(metricas['reporte_clasificacion'])
        
        st.subheader("Matriz de Confusión")
        fig = px.imshow(
            metricas['matriz_confusion'],
            labels=dict(x="Predicho", y="Real", color="Cantidad"),
            x=[f"Clase {i}" for i in range(len(metricas['matriz_confusion']))],
            y=[f"Clase {i}" for i in range(len(metricas['matriz_confusion']))],
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:  # regresión
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Error Cuadrático Medio (MSE)", f"{metricas['mse']:.4f}")
        with col2:
            st.metric("Raíz del Error Cuadrático Medio (RMSE)", f"{metricas['rmse']:.4f}")
        with col3:
            st.metric("R² Score", f"{metricas['r2']:.4f}")

# --- Barra lateral ---
st.sidebar.header("Configuración")

# Selector de tema
tema = st.sidebar.selectbox(
    "Tema de la aplicación",
    ["Claro", "Oscuro"],
    index=0
)

# Aplicar tema
if tema == "Oscuro":
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sección de carga de datos ---
st.sidebar.header("1. Cargar Datos")

# Opción para subir archivo local o usar un enlace de Backblaze
tipo_carga = st.sidebar.radio(
    "¿Cómo deseas cargar tus datos?",
    ["Subir archivo local", "Usar enlace de Backblaze"]
)

archivo_cargado = None
if tipo_carga == "Subir archivo local":
    archivo_cargado = st.sidebar.file_uploader(
        "Sube tu archivo de datos",
        type=["csv", "xlsx", "xls"],
        help="Formatos soportados: CSV, Excel (XLSX, XLS)"
    )
else:
    enlace_backblaze = st.sidebar.text_input("Ingresa la URL de tu archivo en Backblaze")
    if enlace_backblaze:
        try:
            # Extraer el nombre del archivo de la URL
            nombre_archivo = enlace_backblaze.split('/')[-1]
            ruta_local = UPLOAD_DIR / nombre_archivo
            
            # Descargar archivo desde Backblaze
            backblaze.download_file(
                object_name=nombre_archivo,
                local_path=ruta_local,
                bucket_name=B2_BUCKET_INPUT
            )
            
            archivo_cargado = ruta_local
            st.sidebar.success(f"Archivo descargado: {nombre_archivo}")
            
        except Exception as e:
            st.sidebar.error(f"Error al descargar el archivo: {e}")

# Procesar archivo cargado
df = None
if archivo_cargado is not None:
    if hasattr(archivo_cargado, 'name'):  # Si es un archivo subido
        # Guardar archivo localmente
        ruta_archivo = UPLOAD_DIR / archivo_cargado.name
        with open(ruta_archivo, "wb") as f:
            f.write(archivo_cargado.getbuffer())
        
        # Subir a Backblaze
        try:
            enlace_backblaze = backblaze.upload_file(
                ruta_archivo, 
                B2_BUCKET_INPUT, 
                archivo_cargado.name
            )
            st.sidebar.success(f"Archivo subido a Backblaze: {archivo_cargado.name}")
        except Exception as e:
            st.sidebar.warning(f"No se pudo subir a Backblaze: {e}")
        
        # Cargar datos
        df = cargar_datos(archivo_cargado)
    else:  # Si es una ruta local
        df = cargar_datos(archivo_cargado)
    
    if df is not None:
        st.sidebar.success(f"Archivo cargado: {archivo_cargado.name if hasattr(archivo_cargado, 'name') else archivo_cargado}")
        st.sidebar.write(f"- Filas: {len(df):,}")
        st.sidebar.write(f"- Columnas: {len(df.columns)}")
        
        # Inicializar analizador
        st.session_state.analizador = AnalizadorDatos(df)
        analisis = st.session_state.analizador.analizar_dataset()
        
        # Mostrar resumen en la barra lateral
        st.sidebar.subheader("Resumen del Análisis")
        st.sidebar.write(f"- Columnas numéricas: {len(analisis['columnas_numericas'])}")
        st.sidebar.write(f"- Columnas categóricas: {len(analisis['columnas_categoricas'])}")
        
        if analisis['valores_faltantes']['total'] > 0:
            st.sidebar.warning(f"⚠️ {analisis['valores_faltantes']['total']} valores faltantes detectados")

# --- Sección principal ---
if df is not None and st.session_state.analizador:
    # Pestañas para diferentes vistas
    tab1, tab2, tab3 = st.tabs(["📋 Datos", "🔍 Análisis", "🚀 Modelado"])
    
    with tab1:
        st.header("Vista previa de los datos")
        st.dataframe(df.head())
        
        # Mostrar información del dataset
        with st.expander("📋 Información detallada del dataset"):
            st.subheader("Tipos de datos")
            st.write(df.dtypes.astype(str))
            
            st.subheader("Valores faltantes")
            faltantes = df.isnull().sum()
            if faltantes.sum() > 0:
                fig = px.bar(
                    faltantes[faltantes > 0],
                    title="Valores faltantes por columna",
                    labels={'index': 'Columna', 'value': 'Cantidad de valores faltantes'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("¡No hay valores faltantes en el dataset!")
    
    with tab2:
        st.header("Análisis Exploratorio")
        
        if not st.session_state.analisis_completado:
            with st.spinner("Analizando datos..."):
                analisis = st.session_state.analizador.analizar_dataset()
                
                st.subheader("Resumen del Análisis")
                
                # Mostrar sugerencias
                if analisis['sugerencias']:
                    st.warning("Recomendaciones:")
                    for sugerencia in analisis['sugerencias']:
                        with st.expander(f"ℹ️ {sugerencia['mensaje']}"):
                            st.write(sugerencia['accion'])
                
                # Mostrar estadísticas descriptivas
                st.subheader("Estadísticas Descriptivas")
                st.dataframe(df.describe())
                
                # Visualización de correlación para columnas numéricas
                if len(analisis['columnas_numericas']) > 1:
                    st.subheader("Matriz de Correlación")
                    corr = df[analisis['columnas_numericas']].corr()
                    fig = px.imshow(
                        corr,
                        labels=dict(color="Correlación"),
                        x=corr.columns,
                        y=corr.columns,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        zmin=-1,
                        zmax=1
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.analisis_completado = True
    
    with tab3:
        st.header("Modelado Predictivo")
        
        if not st.session_state.analizador:
            st.warning("Por favor, carga un conjunto de datos primero.")
        else:
            # Seleccionar columna objetivo
            columna_objetivo = st.selectbox(
                "Selecciona la variable objetivo (target)",
                df.columns,
                key="target_selector"
            )
            
            if st.button("Iniciar Modelado"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        # Preparar datos
                        X_train, X_test, y_train, y_test = st.session_state.analizador.preparar_datos(columna_objetivo)
                        
                        # Entrenar modelo
                        st.session_state.analizador.entrenar_modelo(X_train, y_train)
                        
                        # Evaluar modelo
                        metricas = st.session_state.analizador.evaluar_modelo(X_test, y_test)
                        
                        # Mostrar resultados
                        st.success("¡Modelo entrenado exitosamente!")
                        mostrar_metricas(metricas, st.session_state.analizador.tipo_problema)
                        
                        # Guardar resultados
                        nombre_archivo = archivo_cargado.name if hasattr(archivo_cargado, 'name') else str(archivo_cargado)
                        email_cliente = st.text_input("Ingresa tu correo electrónico para recibir los resultados")
                        
                        if email_cliente and st.button("Guardar y Enviar Resultados"):
                            with st.spinner("Guardando resultados..."):
                                enlace = guardar_resultados(
                                    st.session_state.analizador,
                                    nombre_archivo,
                                    email_cliente
                                )
                                
                                if enlace:
                                    st.session_state.enlace_resultados = enlace
                                    
                                    # Enviar notificación por correo
                                    if notificador.notificar_analisis_completado(
                                        destinatario=email_cliente,
                                        nombre_cliente="Cliente",
                                        enlace_resultados=enlace,
                                        resumen_analisis=f"Tipo de problema: {st.session_state.analizador.tipo_problema}"
                                    ):
                                        st.success("¡Resultados guardados y notificación enviada!")
                                    else:
                                        st.warning("Los resultados se guardaron, pero hubo un error al enviar la notificación.")
                                        st.write(f"Puedes acceder a tus resultados en: {enlace}")
                    
                    except Exception as e:
                        st.error(f"Error durante el modelado: {str(e)}")
                        
                        # Enviar notificación de error si hay un correo
                        if 'email_cliente' in locals() and email_cliente:
                            notificador.notificar_error(
                                destinatario=email_cliente,
                                nombre_cliente="Cliente",
                                mensaje_error="Ocurrió un error durante el análisis de tus datos.",
                                detalles_tecnicos=str(e)
                            )

# Mensaje de bienvenida si no hay datos cargados
else:
    st.markdown("""
    ## 👋 ¡Bienvenido al Analizador de Datos Automatizado!
    
    Esta herramienta te permite analizar tus datos de forma automática y obtener información valiosa
    sin necesidad de programar.
    
    ### ¿Cómo funciona?
    1. **Sube tus datos** usando el panel de la izquierda (CSV o Excel)
    2. Explora y analiza tus datos en las diferentes pestañas
    3. Entrena un modelo predictivo con un solo clic
    4. Recibe los resultados por correo electrónico
    
    ### Características principales:
    - Análisis exploratorio automático
    - Detección de valores atípicos y faltantes
    - Modelado predictivo (clasificación y regresión)
    - Visualizaciones interactivas
    - Almacenamiento seguro en la nube
    
    ---
    
    *Desarrollado con ❤️ por tu equipo de análisis de datos*
    """)
    
    # Mostrar ejemplo de datos si el usuario lo desea
    if st.checkbox("¿Quieres ver un ejemplo?"):
        st.subheader("Ejemplo de datos")
        datos_ejemplo = pd.DataFrame({
            'edad': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'ingresos': [40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000],
            'gasto_mensual': [3500, 3800, 4200, 4500, 4800, 5000, 5200, 5500, 5800, 6000],
            'compro_producto': ['No', 'No', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí']
        })
        st.dataframe(datos_ejemplo)
        
        st.write("""
        Este es un ejemplo de cómo podrían verse tus datos. Para comenzar con tu propio análisis,
        simplemente sube tu archivo usando el panel de la izquierda.
        """)    key="target_col"
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
