"""
Ejemplo Simple de Análisis de Datos
----------------------------------
Este es un ejemplo simplificado del flujo de trabajo de análisis de datos.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def cargar_datos(ruta_archivo):
    """Carga los datos desde un archivo CSV."""
    try:
        datos = pd.read_csv(ruta_archivo)
        print(f"✅ Datos cargados correctamente. Filas: {len(datos)}")
        return datos
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")
        return None

def preparar_datos(datos, objetivo):
    """Prepara los datos para el modelo."""
    # Eliminar filas con valores faltantes
    datos = datos.dropna()
    
    # Separar características (X) y variable objetivo (y)
    X = datos.drop(columns=[objetivo])
    y = datos[objetivo]
    
    # Convertir variables categóricas a numéricas si es necesario
    X = pd.get_dummies(X)
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

def entrenar_modelo(X_entrenamiento, y_entrenamiento):
    """Entrena un modelo de Random Forest simple."""
    print("🔧 Entrenando modelo...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_entrenamiento, y_entrenamiento)
    print("✅ Modelo entrenado con éxito")
    return modelo

def evaluar_modelo(modelo, X_prueba, y_prueba):
    """Evalúa el rendimiento del modelo."""
    predicciones = modelo.predict(X_prueba)
    precision = accuracy_score(y_prueba, predicciones)
    print(f"📊 Precisión del modelo: {precision:.2f}")
    return precision

def visualizar_importancias(modelo, caracteristicas):
    """Muestra un gráfico de las características más importantes."""
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Importancia de las características")
    plt.bar(range(X_entrenamiento.shape[1]), importancias[indices])
    plt.xticks(range(X_entrenamiento.shape[1]), 
              [caracteristicas[i] for i in indices], 
              rotation=90)
    plt.tight_layout()
    plt.savefig('importancia_caracteristicas.png')
    print("📈 Gráfico de importancia guardado como 'importancia_caracteristicas.png'")

if __name__ == "__main__":
    # Configuración
    RUTA_DATOS = "datos_ejemplo.csv"  # Cambia esto por la ruta a tus datos
    VARIABLE_OBJETIVO = "objetivo"    # Cambia esto por el nombre de tu columna objetivo
    
    print("🚀 Iniciando análisis de datos...")
    
    # 1. Cargar datos
    datos = cargar_datos(RUTA_DATOS)
    if datos is None:
        exit(1)
    
    # 2. Preparar datos
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = preparar_datos(datos, VARIABLE_OBJETIVO)
    
    # 3. Entrenar modelo
    modelo = entrenar_modelo(X_entrenamiento, y_entrenamiento)
    
    # 4. Evaluar modelo
    evaluar_modelo(modelo, X_prueba, y_prueba)
    
    # 5. Visualizar resultados
    visualizar_importancias(modelo, X_entrenamiento.columns)
    
    print("✨ Análisis completado con éxito")
