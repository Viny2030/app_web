# ===============================================================
# MÓDULO datos_auditoria.py
# Contiene la lógica para generar datos y aplicar reglas de auditoría
# ===============================================================

import pandas as pd
import numpy as np
from faker import Faker
import random
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

fake = Faker('es_AR')


def generar_datos_simulados(num_procesos=200):
    """Genera un DataFrame simulado de productos en proceso (WIP)."""

    productos = ['Producto A', 'Producto B', 'Producto C', 'Producto D']
    etapas = ['Corte', 'Ensamblaje', 'Pintura', 'Control de Calidad', 'Empaque']
    lineas = ['Línea 1', 'Línea 2', 'Línea 3']
    estados = ['Activo', 'Pausado', 'Pendiente']

    data = {
        'id_proceso': [fake.uuid4()[:8] for _ in range(num_procesos)],
        'producto': np.random.choice(productos, num_procesos),
        'etapa_actual': np.random.choice(etapas, num_procesos),
        'linea_produccion': np.random.choice(lineas, num_procesos),
        'cantidad_en_proceso': np.random.randint(10, 1000, num_procesos),
        'avance_porcentaje': np.random.uniform(5, 99, num_procesos),
        'estado': np.random.choice(estados, num_procesos),
        'fecha_inicio': [fake.date_between(start_date='-60d', end_date='today') for _ in range(num_procesos)]
    }

    df = pd.DataFrame(data)

    # Introducir algunas anomalías simuladas para la auditoría (baseline)
    df.loc[
        df['etapa_actual'].isin(['Pintura', 'Control de Calidad']) & (df['avance_porcentaje'] < 20), 'is_anomaly'] = 1
    df.loc[(df['linea_produccion'] == 'Línea 3') & (df['cantidad_en_proceso'] > 800), 'is_anomaly'] = 1
    df.loc[(df['estado'] == 'Pausado') & (df['avance_porcentaje'] > 90), 'is_anomaly'] = 1

    df['is_anomaly'] = df['is_anomaly'].fillna(0).astype(int)
    return df


def detectar_outliers_iqr(df, columna):
    """Detecta outliers usando el método IQR (Interquartile Range)."""
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[columna] < lower_bound) | (df[columna] > upper_bound)


def detectar_outliers_zscore(df, columna, threshold=3):
    """Detecta outliers usando Z-score."""
    z_scores = np.abs(stats.zscore(df[columna].fillna(df[columna].mean())))
    return z_scores > threshold


def calcular_scoring_riesgo(df):
    """Calcula un score de riesgo (0-100) para cada proceso basado en múltiples factores."""
    df = df.copy()
    score = np.zeros(len(df))
    
    # Factor 1: Avance muy bajo (0-30 puntos)
    avance_bajo = df['avance_porcentaje'] < 20
    score += np.where(avance_bajo, 30, 0)
    score += np.where((df['avance_porcentaje'] >= 20) & (df['avance_porcentaje'] < 40), 15, 0)
    
    # Factor 2: Estado pausado o pendiente (0-20 puntos)
    score += np.where(df['estado'] == 'Pausado', 20, 0)
    score += np.where(df['estado'] == 'Pendiente', 10, 0)
    
    # Factor 3: Cantidad anormal (0-25 puntos)
    cantidad_media = df['cantidad_en_proceso'].mean()
    cantidad_std = df['cantidad_en_proceso'].std()
    cantidad_anormal = (df['cantidad_en_proceso'] < cantidad_media - 2*cantidad_std) | \
                       (df['cantidad_en_proceso'] > cantidad_media + 2*cantidad_std)
    score += np.where(cantidad_anormal, 25, 0)
    
    # Factor 4: Outliers estadísticos (0-25 puntos)
    outliers_avance = detectar_outliers_iqr(df, 'avance_porcentaje')
    outliers_cantidad = detectar_outliers_iqr(df, 'cantidad_en_proceso')
    score += np.where(outliers_avance | outliers_cantidad, 25, 0)
    
    # Normalizar a 0-100
    score = np.clip(score, 0, 100)
    return score


def aplicar_clustering(df, n_clusters=3):
    """Aplica clustering K-means para identificar patrones similares."""
    df = df.copy()
    
    # Seleccionar características numéricas
    features = ['avance_porcentaje', 'cantidad_en_proceso']
    X = df[features].fillna(df[features].mean())
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    return df, kmeans


def analizar_correlaciones(df):
    """Calcula matriz de correlaciones entre variables numéricas."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].corr()


def detectar_procesos_estancados(df):
    """Detecta procesos que podrían estar estancados basado en avance y estado."""
    estancados = (df['avance_porcentaje'] < 30) & \
                 (df['estado'] == 'Activo') & \
                 (df['cantidad_en_proceso'] > 100)
    return estancados


def detectar_desviaciones_linea(df):
    """Detecta desviaciones significativas por línea de producción."""
    desviaciones = []
    for linea in df['linea_produccion'].unique():
        df_linea = df[df['linea_produccion'] == linea]
        avance_medio_linea = df_linea['avance_porcentaje'].mean()
        avance_medio_global = df['avance_porcentaje'].mean()
        
        # Si la línea se desvía más del 20% del promedio global
        if abs(avance_medio_linea - avance_medio_global) > 20:
            desviaciones.append(linea)
    
    return df['linea_produccion'].isin(desviaciones)


def analizar_eficiencia_etapa(df):
    """Analiza la eficiencia promedio por etapa."""
    eficiencia = df.groupby('etapa_actual').agg({
        'avance_porcentaje': 'mean',
        'cantidad_en_proceso': 'mean',
        'id_proceso': 'count'
    }).rename(columns={'id_proceso': 'total_procesos'})
    eficiencia['eficiencia_score'] = (eficiencia['avance_porcentaje'] / 100) * \
                                     (eficiencia['cantidad_en_proceso'] / eficiencia['cantidad_en_proceso'].max())
    return eficiencia


def aplicar_auditoria_interactiva(df, umbral_min_avance, cantidad_min_ensamblaje):
    """Aplica reglas heurísticas usando parámetros interactivos y algoritmos avanzados."""

    required_cols = ['cantidad_en_proceso', 'avance_porcentaje', 'etapa_actual', 'estado', 'producto']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    df = df.copy()
    df['alerta_heuristica'] = "Sin alertas"

    # Regla 1 (INTERACTIVA): Avance lento para Producto A
    df.loc[(df['producto'] == 'Producto A') &
           (df['avance_porcentaje'] < umbral_min_avance) &
           (df['estado'] == 'Activo'),
    'alerta_heuristica'] = f"Avance lento (< {umbral_min_avance}%)"

    # Regla 2: Procesos finalizados (avance 100%) pero en etapa intermedia
    df.loc[(df['avance_porcentaje'] >= 99.9) & (~df['etapa_actual'].isin(['Empaque', 'Control de Calidad'])),
    'alerta_heuristica'] = "Avance al 100% en etapa intermedia"

    # Regla 3 (INTERACTIVA): Cantidad en proceso demasiado baja para iniciar Ensamblaje
    df.loc[(df['etapa_actual'] == 'Ensamblaje') & (df['cantidad_en_proceso'] < cantidad_min_ensamblaje),
    'alerta_heuristica'] = f"Baja cantidad (< {cantidad_min_ensamblaje}) para Ensamblaje"

    # Regla 4: Procesos estancados
    procesos_estancados = detectar_procesos_estancados(df)
    df.loc[procesos_estancados & (df['alerta_heuristica'] == "Sin alertas"),
    'alerta_heuristica'] = "Proceso posiblemente estancado"

    # Regla 5: Desviaciones por línea de producción
    desviaciones_linea = detectar_desviaciones_linea(df)
    df.loc[desviaciones_linea & (df['alerta_heuristica'] == "Sin alertas"),
    'alerta_heuristica'] = "Desviación significativa en línea de producción"

    # Regla 6: Outliers estadísticos
    outliers_avance = detectar_outliers_iqr(df, 'avance_porcentaje')
    outliers_cantidad = detectar_outliers_iqr(df, 'cantidad_en_proceso')
    df.loc[(outliers_avance | outliers_cantidad) & (df['alerta_heuristica'] == "Sin alertas"),
    'alerta_heuristica'] = "Valor atípico detectado (IQR)"

    # Calcular scoring de riesgo
    df['score_riesgo'] = calcular_scoring_riesgo(df)
    
    # Aplicar clustering
    df, kmeans_model = aplicar_clustering(df, n_clusters=3)
    
    # Detectar outliers con Z-score
    df['outlier_zscore_avance'] = detectar_outliers_zscore(df, 'avance_porcentaje')
    df['outlier_zscore_cantidad'] = detectar_outliers_zscore(df, 'cantidad_en_proceso')
    df['es_outlier'] = df['outlier_zscore_avance'] | df['outlier_zscore_cantidad']

    # --- Resultado Final ---

    # Marcamos como Anómalo si la heurística o la anomalía simulada (is_anomaly) se activa
    if 'is_anomaly' not in df.columns:
        df['is_anomaly'] = 0

    df['resultado_auditoria'] = np.where(
        (df['is_anomaly'] == 1) | (df['alerta_heuristica'] != "Sin alertas") | (df['es_outlier']),
        "Anómalo", "Normal")

    return df