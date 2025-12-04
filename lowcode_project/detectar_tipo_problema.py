import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


def detectar_tipo_problema(df: pd.DataFrame):
    """
    Detecta el tipo de problema (clasificación, regresión, clustering, series_temporales)
    y devuelve el nombre de la columna objetivo si es relevante.

    Retorna: (str: tipo_problema, str | None: nombre_target)
    """

    # 1) Detectar si es una serie temporal 🕰️
    # Comprobamos si el índice o alguna columna es de tipo fecha/tiempo.
    if df.index.dtype == "datetime64[ns]" or any(is_datetime64_any_dtype(df[col]) for col in df.columns):
        # En series temporales, el target suele ser la última columna numérica,
        # pero para el flujo general, no lo forzamos.
        return "series_temporales", None

    # 2) Detectar si existe target (Problema Supervisado) 🎯
    # Lista de nombres comunes para la variable objetivo
    posibles_targets = ["target", "label", "objetivo", "clase"]
    nombre_target = None

    for col in df.columns:
        # Hacemos la comparación en minúsculas para ser insensible a mayúsculas/minúsculas
        if col.lower() in posibles_targets:
            nombre_target = col
            break

    if nombre_target:
        # Clasificación (Target categórico/texto) vs Regresión (Target numérico)

        # Eliminar filas con valores faltantes en la columna objetivo antes de la verificación
        # Esto previene errores de dtype si solo hay nulos en la columna.
        target_series = df[nombre_target].dropna()

        # Si el target no es numérico, lo consideramos clasificación
        if not is_numeric_dtype(target_series):
            return "clasificacion", nombre_target

        # Si el target es numérico, lo consideramos regresión
        else:
            return "regresion", nombre_target

    # 3) Si no hay target → Clustering o Anomalías (Problema No Supervisado) 🧩
    return "clustering_o_anomalias", None