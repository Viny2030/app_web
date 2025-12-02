# test_run.py

import os
import shutil
import subprocess
from worker_proceso_automatico import procesar_dataset
# 💡 Asumimos que la función de subida está implementada en b2_integrator.py
from b2_integrator import subir_archivo_b2

# --- CONFIGURACIÓN DE PRUEBA ---
# ⚠️ ¡IMPORTANTE! Asegúrate de que 'dataset_inversiones_prueba.csv' exista en el bucket 'dataset-raw'
NOMBRE_ARCHIVO_PRUEBA = "dataset_inversiones_prueba.csv"
OUTPUT_DIR = "reporte_generado"

# 🔑 CONFIGURACIÓN DE BACKBLAZE B2:
# 1. ACTUALIZADO: Usamos 'repositorio-web' que es tu bucket real de resultados.
B2_BUCKET_RESULTADOS = "repositorio-web"
# 2. ACTUALIZADO: La URL base debe reflejar el nombre del nuevo bucket.
BASE_URL_CLIENTE = f"https://f005.backblazeb2.com/file/{B2_BUCKET_RESULTADOS}"


# --- FUNCIÓN DE DESPLIEGUE (Reemplaza la ejecución Docker local) ---
def desplegar_dashboard_a_b2(ruta_zip, nombre_archivo_b2):
    """
    Sube el archivo ZIP del dashboard al bucket de resultados de Backblaze B2.
    """
    print("\n--- Despliegue a Backblaze B2 ---")

    # 1. Definir el nombre de la Key para el archivo de resultados
    nombre_key_b2 = os.path.basename(ruta_zip)

    try:
        # 2. Subir el archivo ZIP al segundo bucket
        subir_archivo_b2(
            ruta_local=ruta_zip,
            nombre_bucket=B2_BUCKET_RESULTADOS,
            key_b2=nombre_key_b2
        )
        print(f"✅ Archivo '{nombre_key_b2}' subido exitosamente al bucket '{B2_BUCKET_RESULTADOS}'.")

        # 3. Generar el enlace final para el cliente
        link_acceso = f"{BASE_URL_CLIENTE}/{nombre_key_b2}"
        return link_acceso

    except Exception as e:
        print(f"❌ ERROR durante el despliegue a Backblaze B2: {e}")
        # Muestra el detalle completo del error de B2
        print(f"   Detalle completo: {e}")
        return None


# test_run.py

# ... (código previo) ...

# --- FLUJO PRINCIPAL ---
if __name__ == "__main__":

    # Limpiar ejecuciones anteriores
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    print(f"Iniciando proceso para el archivo de Backblaze: {NOMBRE_ARCHIVO_PRUEBA}")

    # Inicialización de variables para prevenir KeyError si 'procesar_dataset' falla
    ruta_zip, modelo, resultados = None, None, None

    try:
        # 1. Proceso ML y generación del ZIP
        ruta_zip, modelo, resultados = procesar_dataset(NOMBRE_ARCHIVO_PRUEBA, OUTPUT_DIR)

        print("\n=== Resumen del Proceso ML ===")
        # Intentamos obtener el tipo de problema del diccionario de resultados si está disponible
        # 💡 CORRECCIÓN: Usamos .get() de forma segura y solo si 'resultados' existe
        tipo_problema = resultados.get('tipo_problema', 'Desconocido') if resultados else 'Fallo_en_ML'
        print(f"Tipo de problema: {tipo_problema}")
        print(f"Mejor Modelo: {type(modelo).__name__ if not isinstance(modelo, str) else modelo}")
        print(f"Resultados/Score: {resultados}")

        # 2. Pasar a la etapa de despliegue en B2
        # Solo intentamos desplegar si el proceso anterior no fue nulo
        if ruta_zip:
            link_cliente = desplegar_dashboard_a_b2(ruta_zip, NOMBRE_ARCHIVO_PRUEBA)
        else:
            link_cliente = None
            print("\n🚨 El proceso ML falló antes de generar el ZIP. No hay nada que desplegar.")


        if link_cliente:
            print("\n==============================================")
            print("✅ PROCESO COMPLETO Y DESPLIEGUE FINALIZADO ✅")
            print(f"🔗 **Link para el Cliente (Paso 7)**: {link_cliente}")
            print("==============================================")
        else:
            print("\n🚨 El despliegue a Backblaze B2 falló. Verifica las credenciales B2 en 'b2_integrator.py' o el error en el paso ML.")

    except Exception as e:
        print(f"\n❌ Proceso fallido en la etapa ML/B2 (Error General): {e}")