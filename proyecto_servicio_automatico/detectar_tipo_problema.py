import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

def detectar_tipo_problema(df: pd.DataFrame):
    """
    Detecta si el dataset es:
    - clasificación
    - regresión
    - clustering/anomalías
    - series temporales
    """

    # 1) Detectar si es una serie temporal
    if df.index.dtype == "datetime64[ns]" or any(is_datetime64_any_dtype(df[col]) for col in df.columns):
        return "series_temporales"

    # 2) Detectar si existe target
    posibles_targets = ["target", "label", "objetivo", "clase"]
    target = None

    for col in df.columns:
        if col.lower() in posibles_targets:
            target = col
            break

    if target:
        # Clasificación vs regresión
        if not is_numeric_dtype(df[target]):
            return "clasificacion"
        else:
            return "regresion"

    # 3) Si no hay target → clustering o anomalías
    return "clustering_o_anomalias"
