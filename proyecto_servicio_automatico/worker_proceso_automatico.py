import pandas as pd
# 💡 Importar la función de B2
from b2_integrator import descargar_y_leer_csv_b2

from detectar_tipo_problema import detectar_tipo_problema
from selector_modelo import seleccionar_modelos
from entrenador_automatico import entrenar_y_seleccionar
from generar_dashboard import generar_dashboard


# 💡 La función ahora debe llamarse diferente o aceptar el nombre del archivo B2
# Usaremos 'nombre_archivo_b2' como entrada
def procesar_dataset(nombre_archivo_b2, output_folder):
    # 1. Leer dataset (¡Usando la función de B2!)
    # Si la descarga falla, 'descargar_y_leer_csv_b2' lanzará una excepción.
    df = descargar_y_leer_csv_b2(nombre_archivo_b2)

    # 2. Detectar tipo de problema
    tipo = detectar_tipo_problema(df)

    # 3. Seleccionar modelos candidatos
    modelos = seleccionar_modelos(tipo)

    mejor_modelo = None
    pred = []
    reales = []
    resultados = {"tipo_problema": tipo}  # Agregamos el tipo al diccionario de resultados

    # 4. Preparar datos y entrenar
    if tipo in ["clasificacion", "regresion"]:

        # ⚠️ Nota: Asume que la columna target se llama "target"
        if "target" not in df.columns:
            # Manejo de error o caída a clustering/anomalías si no se encuentra el target esperado
            print("Alerta: 'target' no encontrado. Cayendo a Clustering.")
            tipo = "clustering_o_anomalias"
            resultados["tipo_problema"] = tipo  # Actualizar el tipo
        else:
            y = df["target"]
            X = df.drop("target", axis=1)

            # =======================================================
            # 💡 SOLUCIÓN: LIMPIEZA DE COLUMNAS NO NUMÉRICAS
            # =======================================================
            print("-> Limpiando características no numéricas (solo se mantienen tipos numéricos para ML).")

            # Identificar columnas no numéricas en las características (X)
            # 'object' incluye la mayoría de las strings que causan el error
            columnas_a_eliminar = X.select_dtypes(include=['object', 'category']).columns

            X = X.drop(columns=columnas_a_eliminar, errors='ignore')

            print(f"-> Columnas eliminadas en X: {columnas_a_eliminar.tolist()}")
            # =======================================================

            mejor_modelo, resultados_score = entrenar_y_seleccionar(X, y, modelos, tipo)

            # Combinar resultados
            resultados.update(resultados_score)

            # Generar predicciones y reales para el dashboard
            pred = mejor_modelo.predict(X).tolist()
            reales = y.tolist()

    if tipo == "series_temporales":
        # Manejo de Series Temporales (Pendiente de implementar el entrenamiento real)
        mejor_modelo = "prophet"
        pred = [0] * len(df)
        reales = [1] * len(df)
        resultados.update({"info": "Entrenamiento con Series Temporales (Placeholder)"})

    elif tipo == "clustering_o_anomalias":
        # clustering/anomalías
        # Nos aseguramos de que solo trabajamos con datos numéricos para KMeans/DBSCAN
        X_cluster = df.select_dtypes(include=['number']).fillna(0)  # Rellenar NaNs para modelos no robustos

        # Usamos KMeans como modelo de ejemplo
        modelo = modelos["kmeans"]
        modelo.fit(X_cluster)

        pred = modelo.labels_.tolist()
        reales = [None] * len(pred)  # Reales no aplicables en clustering
        mejor_modelo = modelo
        resultados.update({"clusters_encontrados": modelo.n_clusters})

    # 5. Generar dashboard en streamlit
    ruta_zip = generar_dashboard(
        df=df,
        predicciones=pred,
        reales=reales,
        tipo_problema=tipo,
        resultados_dict=resultados,
        output_dir=output_folder
    )

    return ruta_zip, mejor_modelo, resultados