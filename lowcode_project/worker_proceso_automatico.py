import pandas as pd
import numpy as np
from b2_integrator import descargar_y_leer_csv_b2
from detectar_tipo_problema import detectar_tipo_problema
from selector_modelo import seleccionar_modelos
from entrenador_automatico import entrenar_y_seleccionar
from generar_dashboard import generar_dashboard
import os


# ==============================================================================
# 💡 Función de Preprocesamiento Básico Centralizado
# ==============================================================================
def preprocesar_datos_simples(df: pd.DataFrame, target_name: str | None):
    """
    Realiza un preprocesamiento básico (codificación y manejo de nulos)
    para hacer el DataFrame compatible con scikit-learn.
    """
    df_procesado = df.copy()

    # 1. Imputación simple de nulos: Rellenar nulos con 0 para números, 'missing' para categóricos
    for col in df_procesado.columns:
        if pd.api.types.is_numeric_dtype(df_procesado[col]):
            df_procesado[col] = df_procesado[col].fillna(0)
        else:
            df_procesado[col] = df_procesado[col].fillna('missing')

    # 2. Codificación One-Hot de variables categóricas
    # Excluir la columna target de la codificación si existe y es categórica
    if target_name and not pd.api.types.is_numeric_dtype(df_procesado[target_name]):
        X = df_procesado.drop(columns=[target_name])
        y = df_procesado[target_name]
        X = pd.get_dummies(X, drop_first=True)
        # La variable 'y' se deja categórica para que 'entrenar_y_seleccionar' la maneje
    else:
        # Si es regresión o no hay target, codificamos todo
        X = pd.get_dummies(df_procesado, drop_first=True)
        y = None  # Se establecerá más adelante si hay target

    # Asegurar que no queden valores NaN después del dummy (por si acaso)
    X = X.fillna(0)

    return X, y


# ==============================================================================
# 🚀 Función de Proceso Automático
# ==============================================================================
def procesar_dataset(nombre_archivo_b2, output_folder):
    try:
        print(f"\n=== Iniciando procesamiento de {nombre_archivo_b2} ===")
        
        # 1. Leer dataset
        print("Descargando archivo desde Backblaze B2...")
        df = descargar_y_leer_csv_b2(nombre_archivo_b2)
        
        # Validar que el DataFrame no esté vacío
        if df is None:
            raise ValueError("El DataFrame devuelto por descargar_y_leer_csv_b2 es None")
        if df.empty:
            raise ValueError(f"El archivo {nombre_archivo_b2} está vacío")
        
        print(f"DataFrame cargado correctamente con {len(df)} filas y {len(df.columns)} columnas")
        
        # 2. Detectar tipo de problema y nombre del target
        print("\nDetectando tipo de problema...")
        tipo_detectado, target_col_name = detectar_tipo_problema(df)
        tipo = tipo_detectado
        print(f"Tipo de problema detectado: {tipo}")
        if target_col_name:
            print(f"Columna objetivo detectada: {target_col_name}")

        # Inicialización de variables
        resultados = {"tipo_problema": tipo}
        mejor_modelo = "Modelo no entrenado"
        pred = []
        reales = []
        df_len = len(df)  # Definir la longitud del DF una vez

        # 3. Flujo de Clasificación/Regresión (Problemas Supervisados)
        if tipo in ["clasificacion", "regresion"]:
            if target_col_name is None:
                print("Alerta: Columna objetivo no detectada. Cambiando a Clustering/Anomalías.")
                tipo = "clustering_o_anomalias"
                resultados["tipo_problema"] = tipo
            else:
                X_completo, y_target_dummy = preprocesar_datos_simples(df, target_col_name)

                try:
                    X = X_completo.drop(target_col_name, axis=1, errors='ignore')
                    y = df[target_col_name]

                    print(f"Iniciando entrenamiento para {tipo.upper()} con target: {target_col_name}")

                    mejor_modelo, resultados_score = entrenar_y_seleccionar(X, y, seleccionar_modelos(tipo), tipo)
                    resultados.update(resultados_score)

                    # Generar predicciones y reales
                    if hasattr(mejor_modelo, 'predict'):
                        pred = mejor_modelo.predict(X).tolist()
                        reales = y.tolist()
                    else:
                        pred = [None] * df_len
                        reales = y.tolist() if target_col_name in df.columns else [None] * df_len

                except Exception as e:
                    print(f"❌ ERROR CRÍTICO en el Entrenamiento de {tipo.upper()}: {e}")
                    mejor_modelo = "Entrenamiento Fallido"
                    pred = [None] * df_len
                    reales = df[target_col_name].tolist() if target_col_name in df.columns else [None] * df_len

        # 4. Flujo de Series Temporales
        elif tipo == "series_temporales":
            mejor_modelo = "prophet"
            pred = [None] * df_len  # Usar la longitud del DF

            # Simplificación de la lógica compleja:
            numeric_cols_df = df.select_dtypes(include='number')
            if numeric_cols_df.shape[1] > 0:
                reales = numeric_cols_df.iloc[:, -1].fillna(0).tolist()
            else:
                reales = [np.nan] * df_len

            resultados.update({"info": "Entrenamiento con Series Temporales (Placeholder)"})

        # 5. Flujo de Clustering (No Supervisado)
        elif tipo == "clustering_o_anomalias":
            print("Iniciando proceso de Clustering/Anomalías...")
            modelos = seleccionar_modelos(tipo)
            X_cluster, _ = preprocesar_datos_simples(df, None)

            if X_cluster.shape[1] >= 2:  # Necesitamos al menos 2 características para clustering
                try:
                    print("Aplicando KMeans para clustering...")
                    modelo = modelos["kmeans"]
                    modelo.fit(X_cluster)
                    pred = modelo.labels_.tolist()
                    reales = [None] * df_len  # No hay valores reales en clustering
                    mejor_modelo = modelo
                    resultados.update({
                        "clusters_encontrados": int(modelo.n_clusters), 
                        "metodo": "KMeans"
                    })
                    print(f"Clustering completado con {modelo.n_clusters} clusters")
                except Exception as e:
                    print(f"❌ ERROR CRÍTICO en el entrenamiento de CLUSTERING: {e}")
                    mejor_modelo = "Clustering Fallido (Ver logs)"
                    pred = [None] * df_len
                    reales = [None] * df_len
                    resultados.update({
                        "error": f"Fallo al entrenar KMeans: {e}", 
                        "status": "Error"
                    })
            else:
                print("Alerta: Menos de 2 columnas válidas para Clustering.")
                mejor_modelo = "Clustering no ejecutado"
                pred = [None] * df_len
                reales = [None] * df_len
                resultados.update({
                    "error": "No hay suficientes datos numéricos o codificados para clustering.",
                    "status": "Error"
                })

        # 6. Generar dashboard
        print("\nGenerando dashboard...")
        try:
            ruta_zip = generar_dashboard(
                df=df,
                predicciones=pred,
                reales=reales,
                tipo_problema=tipo,
                resultados_dict=resultados,
                output_dir=output_folder
            )

            if not ruta_zip or not os.path.exists(ruta_zip):
                raise ValueError(f"No se pudo generar el archivo ZIP en {ruta_zip}")

            print(f"\n✅ Proceso completado exitosamente")
            print(f"📁 Archivo generado: {ruta_zip}")
            return ruta_zip, mejor_modelo, resultados

        except Exception as e:
            print(f"❌ ERROR al generar el dashboard: {e}")
            raise

    except Exception as e:
        print(f"\n❌ ERROR en procesar_dataset: {str(e)}")
        print(f"Tipo de error: {type(e).__name__}")
        import traceback
        print("\nStack trace completo:")
        print(traceback.format_exc())
        raise  # Re-lanzar la excepción para que se vea en el log principal