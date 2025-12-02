import pandas as pd
from b2_integrator import descargar_y_leer_csv_b2
from detectar_tipo_problema import detectar_tipo_problema
from selector_modelo import seleccionar_modelos
from entrenador_automatico import entrenar_y_seleccionar
from generar_dashboard import generar_dashboard
import os


def procesar_dataset(nombre_archivo_b2, output_folder):
    # 1. Leer dataset
    df = descargar_y_leer_csv_b2(nombre_archivo_b2)

    # 2. Detectar tipo de problema
    tipo = detectar_tipo_problema(df)

    # Inicialización de variables (para robustez y evitar errores)
    resultados = {"tipo_problema": tipo}
    mejor_modelo = "Modelo no entrenado"
    pred = []
    reales = []

    # 3. Flujo de Clasificación/Regresión
    if tipo in ["clasificacion", "regresion"]:

        # 3.1. VERIFICACIÓN DEL TARGET
        if "target" not in df.columns:
            # Si se esperaba Clasificación/Regresión pero no hay 'target', cambiamos el tipo.
            print("Alerta: 'target' no encontrado. Cambiando a Clustering/Anomalías.")
            tipo = "clustering_o_anomalias"
            resultados["tipo_problema"] = tipo
            # El flujo continuará en el bloque 'elif tipo == "clustering_o_anomalias"'

        else:
            # EJECUCIÓN CLASIFICACIÓN/REGRESIÓN
            y = df["target"]
            X = df.drop("target", axis=1)

            # Limpieza de columnas no numéricas...
            columnas_a_eliminar = X.select_dtypes(include=['object', 'category']).columns
            X = X.drop(columns=columnas_a_eliminar, errors='ignore')

            # Entrenamiento con Manejo de Errores
            try:
                mejor_modelo, resultados_score = entrenar_y_seleccionar(X, y, seleccionar_modelos(tipo), tipo)
                resultados.update(resultados_score)

                # Generar predicciones para el dashboard
                pred = mejor_modelo.predict(X).tolist()
                reales = y.tolist()

            except Exception as e:
                print(f"❌ ERROR CRÍTICO en el Entrenamiento de {tipo.upper()}: {e}")
                mejor_modelo = "Entrenamiento Fallido"
                pred = [None] * len(y) if 'y' in locals() else []
                reales = y.tolist() if 'y' in locals() else []

    # 4. Flujo de Series Temporales (ALINEADO CORRECTAMENTE)
    elif tipo == "series_temporales":
        mejor_modelo = "prophet"
        pred = [0] * len(df)
        reales = [1] * len(df)
        resultados.update({"info": "Entrenamiento con Series Temporales (Placeholder)"})

    # 5. Flujo de Clustering (ALINEADO CORRECTAMENTE)
    elif tipo == "clustering_o_anomalias":
        modelos = seleccionar_modelos(tipo)
        X_cluster = df.select_dtypes(include=['number']).fillna(0)

        # Se requiere al menos 2 columnas para el gráfico y entrenamiento robusto
        if X_cluster.shape[1] >= 2:

            try:
                modelo = modelos["kmeans"]
                modelo.fit(X_cluster)
                pred = modelo.labels_.tolist()
                reales = [None] * len(pred)
                mejor_modelo = modelo
                resultados.update({"clusters_encontrados": modelo.n_clusters, "metodo": "KMeans"})

            except Exception as e:
                print(f"❌ ERROR CRÍTICO en el entrenamiento de CLUSTERING: {e}")
                mejor_modelo = "Clustering Fallido (Ver logs)"
                pred = [None] * len(X_cluster)
                resultados.update({"error": f"Fallo al entrenar KMeans: {e}", "status": "Error"})

        else:
            print("Alerta: Menos de 2 columnas numéricas para Clustering.")
            mejor_modelo = "Clustering no ejecutado"
            resultados.update({"error": "No hay suficientes datos numéricos para clustering."})

    # 6. Generar dashboard (Asegurar que esta función está fuera de los bloques IF/ELIF)
    ruta_zip = generar_dashboard(
        df=df,
        predicciones=pred,
        reales=reales,
        tipo_problema=tipo,
        resultados_dict=resultados,
        output_dir=output_folder
    )

    return ruta_zip, mejor_modelo, resultados