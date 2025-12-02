from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np


def entrenar_y_seleccionar(X, y, modelos, tipo_problema):
    resultados = {}
    mejor_modelo = None
    mejor_score = -np.inf if tipo_problema == "clasificacion" else np.inf

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for nombre, modelo in modelos.items():

        # Series temporales se manejan afuera
        if tipo_problema == "series_temporales":
            continue

        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

        # Clasificación
        if tipo_problema == "clasificacion":
            score = accuracy_score(y_test, pred)
            resultados[nombre] = score
            if score > mejor_score:
                mejor_score = score
                mejor_modelo = modelo

        # Regresión
        elif tipo_problema == "regresion":
            score = mean_squared_error(y_test, pred)
            resultados[nombre] = score
            if score < mejor_score:
                mejor_score = score
                mejor_modelo = modelo

    return mejor_modelo, resultados
