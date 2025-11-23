import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
from typing import Tuple, List


# --- 1. FUNCIÓN DE GENERACIÓN DE DATOS SIMULADOS ---
def generar_datos_simulados() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Crea DataFrames de movimientos bancarios y contables."""
    fake = Faker('es_AR')
    np.random.seed(123)
    random.seed(123)

    num_transacciones = 100
    tipos_transaccion = ['Débito', 'Crédito']
    conceptos_banco = ['Transferencia recibida', 'Transferencia enviada', 'Pago proveedor', 'Comisión bancaria']
    conceptos_contables = ['Pago proveedor', 'Cobro cliente', 'Ajuste de tipo de cambio']

    def generar_fecha(dias_delta: int = 0):
        return (fake.date_between(start_date='-60d', end_date='today') + timedelta(days=dias_delta))

    movimientos_banco = []
    movimientos_contables = []

    for i in range(num_transacciones):
        fecha = generar_fecha()
        tipo = random.choice(tipos_transaccion)
        monto = round(random.uniform(500, 10000), 2)

        movimientos_banco.append({
            'id_banco': i + 1,
            'fecha': fecha,
            'tipo': tipo,
            'concepto_banco': random.choice(conceptos_banco),
            'monto_banco': monto,
            'referencia_banco': fake.unique.bothify('BNK-########')
        })

        # 85% de chance de tener contrapartida contable
        if random.random() > 0.15:
            monto_contable = monto
            if random.random() > 0.8:  # Introduce diferencia leve en el monto (20%)
                monto_contable = round(monto + random.uniform(-100, 100), 2)

            movimientos_contables.append({
                'id_contable': i + 1,
                'fecha': generar_fecha(random.randint(0, 3)),  # Diferencia de fecha (hasta 3 días)
                'tipo': tipo,
                'concepto_contable': random.choice(conceptos_contables),
                'monto_contable': monto_contable,
                'referencia_contable': fake.unique.bothify('CNT-########')
            })

    return pd.DataFrame(movimientos_banco), pd.DataFrame(movimientos_contables)


# --- 2. FUNCIÓN DE CONCILIACIÓN Y AUDITORÍA ---
def conciliar_y_auditar(df_banco: pd.DataFrame, df_contabilidad: pd.DataFrame):
    TOLERANCIA_MONTO = 50
    TOLERANCIA_DIAS = 5

    # Asegurar que las fechas sean datetime
    df_banco['fecha'] = pd.to_datetime(df_banco['fecha'])
    df_contabilidad['fecha'] = pd.to_datetime(df_contabilidad['fecha'])

    # Outer merge basado solo en el tipo (para auditar todos los posibles emparejamientos)
    df_merged = pd.merge(df_banco, df_contabilidad,
                         left_on='tipo',
                         right_on='tipo',  # Nota: Usamos 'tipo' como clave de merge
                         how='outer',
                         suffixes=('_banco', '_contable'),
                         indicator=True)

    # Convertir NaN a 0 para cálculos y calcular diferencias
    df_merged['monto_banco'] = df_merged['monto_banco'].fillna(0)
    df_merged['monto_contable'] = df_merged['monto_contable'].fillna(0)

    df_merged['diferencia_monto'] = np.abs(df_merged['monto_banco'] - df_merged['monto_contable'])
    df_merged['diferencia_dias'] = np.abs(df_merged['fecha_banco'] - df_merged['fecha_contable']).dt.days

    # Regla de conciliación: Cumple tolerancia de monto Y de días
    df_merged['conciliado'] = (
            (df_merged['diferencia_monto'] <= TOLERANCIA_MONTO) &
            (df_merged['diferencia_dias'] <= TOLERANCIA_DIAS)
    ).fillna(False)

    # Clasificación final para auditoría
    def clasificar_transaccion(row):
        if row['_merge'] == 'left_only':
            return "Sólo Banco"
        elif row['_merge'] == 'right_only':
            return "Sólo Contable"
        elif row['_merge'] == 'both':
            if row['conciliado']:
                if row['diferencia_monto'] < 0.01:
                    return "Conciliado Exacto"
                else:
                    return "Conciliado con Diferencia Ajustable"
            else:
                return "No Conciliado - Revisión Manual"
        return "Error de Clasificación"

    df_merged['clasificación_auditoría'] = df_merged.apply(clasificar_transaccion, axis=1)

    # Seleccionar columnas relevantes para el informe final
    return df_merged[['fecha_banco', 'monto_banco', 'concepto_banco',
                      'fecha_contable', 'monto_contable', 'concepto_contable',
                      'diferencia_monto', 'diferencia_dias', 'conciliado',
                      'clasificación_auditoría']]


# --- 3. FUNCIÓN PRINCIPAL DE EJECUCIÓN (Asegura el guardado) ---
def main_generar_y_procesar():
    df_banco, df_contabilidad = generar_datos_simulados()
    df_conciliado = conciliar_y_auditar(df_banco, df_contabilidad)

    # Nombre de archivo solicitado
    nombre_archivo = 'informe_conciliacion.csv'

    # --- GUARDA EL ARCHIVO LOCALMENTE ---
    df_conciliado.to_csv(nombre_archivo, index=False)

    print(f"✅ Archivo local generado: {nombre_archivo}")
    print("\n--- Muestra de 5 Filas del Informe Final ---")

    # Requiere la librería 'tabulate', debe estar instalada: pip install tabulate
    try:
        print(df_conciliado.head(5).to_markdown(index=False))
    except Exception:
        print(df_conciliado.head(5))

    return nombre_archivo


if __name__ == '__main__':
    main_generar_y_procesar()