from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from xgboost import XGBClassifier, XGBRegressor

def seleccionar_modelos(tipo_problema):
    """
    Devuelve un diccionario de modelos candidatos
    según el tipo de problema.
    """

    if tipo_problema == "clasificacion":
        return {
            "logistic": LogisticRegression(max_iter=200),
            "rf": RandomForestClassifier(),
            "xgb": XGBClassifier()
        }

    if tipo_problema == "regresion":
        return {
            "linear": LinearRegression(),
            "rf": RandomForestRegressor(),
            "xgb": XGBRegressor()
        }

    if tipo_problema == "series_temporales":
        return {
            "prophet": "prophet",
            "arima": "pmdarima"
        }

    # clustering o anomalías
    return {
        "kmeans": KMeans(n_clusters=3),
        "dbscan": DBSCAN(),
        "iso_forest": IsolationForest()
    }
