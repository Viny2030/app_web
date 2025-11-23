# =================================================================
# SCRIPT DE LÓGICA DE DATOS Y AUDITORÍA PARA PRODUCTOS EN PROCESO
# =================================================================

# --- 1. IMPORTACIONES DE LÓGICA ---

import pandas as pd
import numpy as np
import random
from datetime import timedelta, datetime
from faker import Faker
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# 2. GENERACIÓN DE DATOS SIMULADOS
# ===============================================================

# ===============================================================
# 2. GENERACIÓN DE DATOS SIMULADOS (Versión Robusta)
# ===============================================================

def generar_datos_simulados():
    """Genera datos simulados de productos en proceso (Utilizando Faker)."""

    # 1. Definiciones
    FECHA_HOY = datetime(2025, 11, 22).date()  # Fecha fija para la simulación
    LIMITE_SUPERIOR = FECHA_HOY - timedelta(days=5)  # Máximo 5 días de antigüedad
    LIMITE_INFERIOR = FECHA_HOY - timedelta(days=30)  # Mínimo 30 días de antigüedad

    np.random.seed(42)
    random.seed(42)
    fake = Faker('es_AR')
    Faker.seed(42)
    num_registros = 50
    etapas_produccion = ['Corte', 'Ensamblado', 'Soldadura', 'Pintura', 'Control de calidad']
    lineas_produccion = ['Línea A', 'Línea B', 'Línea C']
    productos = ['Silla', 'Mesa', 'Estantería', 'Armario', 'Puerta']

    productos_proceso = []
    for i in range(num_registros):

        # Generación estándar de fechas: pasar solo objetos date
        fecha_inicio = fake.date_between(start_date=LIMITE_INFERIOR, end_date=LIMITE_SUPERIOR)
        duracion_estimada = random.randint(1, 10)
        avance_porcentaje = random.randint(10, 95)

        # Simulación de anomalía (proceso viejo con avance bajo)
        if i == 5:
            # Rango anómalo: 45 a 40 días de antigüedad
            fecha_inicio = fake.date_between(
                start_date=FECHA_HOY - timedelta(days=45),
                end_date=FECHA_HOY - timedelta(days=40)
            )
            avance_porcentaje = 5

        productos_proceso.append({
            'id_proceso': f'WIP-{1000 + i}',
            'producto': random.choice(productos),
            'lote': f'L-{fake.random_int(min=1000, max=9999)}',
            'etapa_actual': random.choice(etapas_produccion),
            'linea_produccion': random.choice(lineas_produccion),
            'cantidad_en_proceso': random.randint(10, 200),
            'fecha_inicio': fecha_inicio,
            'fecha_estim_termino': fecha_inicio + timedelta(days=duracion_estimada),
            'avance_porcentaje': avance_porcentaje,
            'estado': 'En proceso'
        })

    return pd.DataFrame(productos_proceso)


# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_auditoria(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas heurísticas y el modelo de detección de anomalías."""
    df['fecha_inicio'] = pd.to_datetime(df['fecha_inicio'])
    df['fecha_estim_termino'] = pd.to_datetime(df['fecha_estim_termino'])

    def reglas_auditoria(row):
        alertas = []
        hoy = pd.to_datetime("today").date()  # Usamos solo la fecha

        # Regla 1: Avance muy bajo (Menos del 10%)
        if row['avance_porcentaje'] <= 10 and row['estado'] == 'En proceso':
            alertas.append("Avance muy bajo")

        # Regla 2: Fecha estimada vencida
        if row['fecha_estim_termino'].date() < hoy and row['estado'] == 'En proceso':
            alertas.append("Fecha estimada vencida")

        # Regla 3: Cantidad inválida
        if row['cantidad_en_proceso'] <= 0:
            alertas.append("Cantidad inválida")

        # Regla 4: Avance fuera de rango
        if row['avance_porcentaje'] > 100 or row['avance_porcentaje'] < 0:
            alertas.append("Avance fuera de rango")

        return " | ".join(alertas) if alertas else "Sin alertas"

    df['alerta_heuristica'] = df.apply(reglas_auditoria, axis=1)

    # Detección de anomalías ML (Isolation Forest)
    features = ['cantidad_en_proceso', 'avance_porcentaje']
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    df['anomaly'] = modelo.fit_predict(X_scaled)
    df['resultado_auditoria'] = df['anomaly'].map({1: 'Normal', -1: 'Anómalo'})

    return df


# ===============================================================
# 4. EJECUCIÓN Y GUARDADO DEL DATAFRAME (AGREGAR ESTO)
# ===============================================================

if __name__ == "__main__":
    # 1. Generar los datos
    df_proceso = generar_datos_simulados()

    # 2. Aplicar la lógica de auditoría
    df_auditado = aplicar_auditoria(df_proceso)

    # 3. Guardar el DataFrame en un archivo CSV
    nombre_archivo = "reporte_datos_auditados.csv"
    df_auditado.to_csv(nombre_archivo, index=False)

    print(f"✅ DataFrame de auditoría generado y guardado como: {nombre_archivo}")
    print("\nPrimeras 5 filas del DataFrame:")
    print(df_auditado.head())