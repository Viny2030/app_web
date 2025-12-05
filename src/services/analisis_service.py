import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, 
    mean_squared_error, r2_score, confusion_matrix
)
import logging
from pathlib import Path
import joblib

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalizadorDatos:
    """Servicio para analizar datos y seleccionar el mejor modelo"""
    
    def __init__(self, df: pd.DataFrame = None):
        """Inicializa el analizador con un DataFrame opcional"""
        self.df = df
        self.analisis = {}
        self.modelo = None
        self.preprocesador = None
        self.columnas_numericas = []
        self.columnas_categoricas = []
        self.columna_objetivo = None
        self.tipo_problema = None
        
    def analizar_dataset(self, df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analiza un conjunto de datos y devuelve un diccionario con metadatos
        
        Args:
            df: DataFrame a analizar (opcional si se proporcionó en el constructor)
            
        Returns:
            Dict con metadatos del análisis
        """
        if df is not None:
            self.df = df
            
        if self.df is None:
            raise ValueError("No se ha proporcionado ningún DataFrame para analizar")
            
        # Inicializar análisis
        self.analisis = {
            'filas': len(self.df),
            'columnas': len(self.df.columns),
            'columnas_numericas': [],
            'columnas_categoricas': [],
            'valores_faltantes': {},
            'estadisticas_descriptivas': {},
            'sugerencias': []
        }
        
        # Identificar tipos de columnas
        self._identificar_tipos_columnas()
        
        # Analizar valores faltantes
        self._analizar_valores_faltantes()
        
        # Generar estadísticas descriptivas
        self._generar_estadisticas()
        
        # Generar sugerencias
        self._generar_sugerencias()
        
        return self.analisis
    
    def _identificar_tipos_columnas(self):
        """Identifica columnas numéricas y categóricas"""
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.columnas_numericas.append(col)
            else:
                self.columnas_categoricas.append(col)
                
        self.analisis['columnas_numericas'] = self.columnas_numericas
        self.analisis['columnas_categoricas'] = self.columnas_categoricas
    
    def _analizar_valores_faltantes(self):
        """Analiza los valores faltantes en el dataset"""
        faltantes = self.df.isnull().sum()
        faltantes_porcentaje = (faltantes / len(self.df)) * 100
        
        self.analisis['valores_faltantes'] = {
            'total': faltantes.sum(),
            'por_columna': faltantes[faltantes > 0].to_dict(),
            'porcentaje_por_columna': faltantes_porcentaje[faltantes_porcentaje > 0].to_dict()
        }
    
    def _generar_estadisticas(self):
        """Genera estadísticas descriptivas para las columnas numéricas"""
        if self.columnas_numericas:
            self.analisis['estadisticas_descriptivas'] = self.df[self.columnas_numericas].describe().to_dict()
    
    def _generar_sugerencias(self):
        """Genera sugerencias basadas en el análisis del dataset"""
        sugerencias = []
        
        # Sugerencias para valores faltantes
        if self.analisis['valores_faltantes']['total'] > 0:
            sugerencias.append({
                'tipo': 'advertencia',
                'mensaje': 'El conjunto de datos contiene valores faltantes que deben ser tratados.',
                'accion': 'Considerar imputación de valores o eliminación de filas/columnas con valores faltantes.'
            })
            
        # Sugerencias para columnas categóricas
        if self.columnas_categoricas:
            sugerencias.append({
                'tipo': 'info',
                'mensaje': f'Se detectaron {len(self.columnas_categoricas)} columnas categóricas.',
                'accion': 'Aplicar codificación one-hot o label encoding antes del modelado.'
            })
            
        self.analisis['sugerencias'] = sugerencias
    
    def determinar_tipo_problema(self, columna_objetivo: str) -> str:
        """
        Determina el tipo de problema (clasificación o regresión) basado en la columna objetivo
        
        Args:
            columna_objetivo: Nombre de la columna objetivo
            
        Returns:
            str: 'clasificacion' o 'regresion'
        """
        self.columna_objetivo = columna_objetivo
        
        if pd.api.types.is_numeric_dtype(self.df[columna_objetivo]):
            # Si hay pocos valores únicos en proporción al total, es clasificación
            valores_unicos = self.df[columna_objetivo].nunique()
            proporcion_unicos = valores_unicos / len(self.df)
            
            if proporcion_unicos < 0.1:  # Umbral arbitrario
                self.tipo_problema = 'clasificacion'
            else:
                self.tipo_problema = 'regresion'
        else:
            self.tipo_problema = 'clasificacion'
            
        return self.tipo_problema
    
    def preparar_datos(self, columna_objetivo: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepara los datos para el modelado
        
        Args:
            columna_objetivo: Nombre de la columna objetivo
            test_size: Proporción del conjunto de prueba
            random_state: Semilla para reproducibilidad
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        if columna_objetivo not in self.df.columns:
            raise ValueError(f"La columna objetivo '{columna_objetivo}' no existe en el DataFrame")
            
        self.columna_objetivo = columna_objetivo
        self.tipo_problema = self.determinar_tipo_problema(columna_objetivo)
        
        # Separar características y objetivo
        X = self.df.drop(columns=[columna_objetivo])
        y = self.df[columna_objetivo]
        
        # Identificar columnas numéricas y categóricas
        columnas_numericas = X.select_dtypes(include=['number']).columns.tolist()
        columnas_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Crear transformadores para preprocesamiento
        transformadores_numericos = Pipeline(steps=[
            ('imputador', SimpleImputer(strategy='median')),
            ('escalador', StandardScaler())
        ])
        
        transformadores_categoricos = Pipeline(steps=[
            ('imputador', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combinar transformadores
        self.preprocesador = ColumnTransformer(
            transformers=[
                ('num', transformadores_numericos, columnas_numericas),
                ('cat', transformadores_categoricos, columnas_categoricas)
            ])
        
        # Aplicar transformaciones
        X_transformado = self.preprocesador.fit_transform(X)
        
        # Dividir en conjuntos de entrenamiento y prueba
        return train_test_split(
            X_transformado, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y if self.tipo_problema == 'clasificacion' else None
        )
    
    def entrenar_modelo(self, X_train, y_train, nombre_modelo: str = None, tipo_problema: str = None, **kwargs):
        """
        Entrena un modelo de machine learning
        
        Args:
            X_train: Características de entrenamiento
            y_train: Variable objetivo de entrenamiento
            nombre_modelo: Nombre del modelo a utilizar. Si es None, se usa RandomForest
            tipo_problema: 'clasificacion' o 'regresion'. Si es None, se determina automáticamente
            **kwargs: Parámetros adicionales para el modelo
            
        Returns:
            Modelo entrenado
        """
        if tipo_problema is None:
            if self.tipo_problema is None:
                raise ValueError("No se ha especificado el tipo de problema. Llame a determinar_tipo_problema() primero.")
            tipo_problema = self.tipo_problema
        
        # Parámetros por defecto
        params = {
            'random_state': kwargs.get('random_state', 42),
            'n_jobs': kwargs.get('n_jobs', -1)
        }
        
        # Seleccionar modelo basado en el nombre y tipo de problema
        if nombre_modelo is None:
            nombre_modelo = 'random_forest'
            
        nombre_modelo = nombre_modelo.lower()
        
        try:
            if tipo_problema == 'clasificacion':
                if nombre_modelo == 'random_forest':
                    self.modelo = RandomForestClassifier(
                        n_estimators=kwargs.get('n_estimators', 100),
                        **params
                    )
                elif nombre_modelo == 'gradient_boosting':
                    self.modelo = GradientBoostingClassifier(
                        n_estimators=kwargs.get('n_estimators', 100),
                        learning_rate=kwargs.get('learning_rate', 0.1),
                        **{k: v for k, v in params.items() if k != 'n_jobs'}
                    )
                elif nombre_modelo == 'svm':
                    self.modelo = SVC(
                        C=kwargs.get('C', 1.0),
                        kernel=kwargs.get('kernel', 'rbf'),
                        probability=True,
                        **{k: v for k, v in params.items() if k != 'n_jobs'}
                    )
                elif nombre_modelo == 'logistic_regression':
                    self.modelo = LogisticRegression(
                        C=kwargs.get('C', 1.0),
                        max_iter=kwargs.get('max_iter', 1000),
                        **params
                    )
                elif nombre_modelo == 'knn':
                    self.modelo = KNeighborsClassifier(
                        n_neighbors=kwargs.get('n_neighbors', 5),
                        **{k: v for k, v in params.items() if k != 'random_state'}
                    )
                elif nombre_modelo == 'xgboost':
                    self.modelo = XGBClassifier(
                        n_estimators=kwargs.get('n_estimators', 100),
                        learning_rate=kwargs.get('learning_rate', 0.1),
                        use_label_encoder=False,
                        eval_metric='logloss',
                        **{k: v for k, v in params.items() if k != 'n_jobs'}
                    )
                elif nombre_modelo == 'lightgbm':
                    self.modelo = LGBMClassifier(
                        n_estimators=kwargs.get('n_estimators', 100),
                        learning_rate=kwargs.get('learning_rate', 0.1),
                        **{k: v for k, v in params.items() if k != 'n_jobs'}
                    )
                elif nombre_modelo == 'catboost':
                    self.modelo = CatBoostClassifier(
                        iterations=kwargs.get('n_estimators', 100),
                        learning_rate=kwargs.get('learning_rate', 0.1),
                        verbose=0,
                        **{k: v for k, v in params.items() if k != 'n_jobs'}
                    )
                else:
                    raise ValueError(f"Modelo de clasificación no soportado: {nombre_modelo}")
                    
            else:  # regresión
                if nombre_modelo == 'random_forest':
                    self.modelo = RandomForestRegressor(
                        n_estimators=kwargs.get('n_estimators', 100),
                        **params
                    )
                elif nombre_modelo == 'gradient_boosting':
                    self.modelo = GradientBoostingRegressor(
                        n_estimators=kwargs.get('n_estimators', 100),
                        learning_rate=kwargs.get('learning_rate', 0.1),
                        **{k: v for k, v in params.items() if k != 'n_jobs'}
                    )
                elif nombre_modelo == 'svm':
                    self.modelo = SVR(
                        C=kwargs.get('C', 1.0),
                        kernel=kwargs.get('kernel', 'rbf'),
                        **{k: v for k, v in params.items() if k != 'n_jobs'}
                    )
                elif nombre_modelo == 'linear_regression':
                    self.modelo = LinearRegression(
                        **{k: v for k, v in params.items() if k != 'random_state'}
                    )
                elif nombre_modelo == 'ridge':
                    self.modelo = Ridge(
                        alpha=kwargs.get('alpha', 1.0),
                        **{k: v for k, v in params.items() if k != 'random_state'}
                    )
                elif nombre_modelo == 'lasso':
                    self.modelo = Lasso(
                        alpha=kwargs.get('alpha', 1.0),
                        **{k: v for k, v in params.items() if k != 'random_state'}
                    )
                elif nombre_modelo == 'xgboost':
                    self.modelo = XGBRegressor(
                        n_estimators=kwargs.get('n_estimators', 100),
                        learning_rate=kwargs.get('learning_rate', 0.1),
                        **{k: v for k, v in params.items() if k != 'n_jobs'}
                    )
                elif nombre_modelo == 'lightgbm':
                    self.modelo = LGBMRegressor(
                        n_estimators=kwargs.get('n_estimators', 100),
                        learning_rate=kwargs.get('learning_rate', 0.1),
                        **{k: v for k, v in params.items() if k != 'n_jobs'}
                    )
                elif nombre_modelo == 'catboost':
                    self.modelo = CatBoostRegressor(
                        iterations=kwargs.get('n_estimators', 100),
                        learning_rate=kwargs.get('learning_rate', 0.1),
                        verbose=0,
                        **{k: v for k, v in params.items() if k != 'n_jobs'}
                    )
                else:
                    raise ValueError(f"Modelo de regresión no soportado: {nombre_modelo}")
            
            # Entrenar el modelo
            self.modelo.fit(X_train, y_train)
            return self.modelo
            
        except Exception as e:
            logger.error(f"Error al entrenar el modelo {nombre_modelo}: {str(e)}")
            raise
    
    def evaluar_modelo(self, X_test, y_test, X_train=None, y_train=None) -> Dict[str, Any]:
        """
        Evalúa el modelo entrenado con métricas detalladas
        
        Args:
            X_test: Características de prueba
            y_test: Variable objetivo de prueba
            X_train: Características de entrenamiento (opcional para métricas adicionales)
            y_train: Variable objetivo de entrenamiento (opcional para métricas adicionales)
            
        Returns:
            Dict con métricas de evaluación detalladas
        """
        if self.modelo is None:
            raise ValueError("No hay ningún modelo entrenado. Entrene un modelo primero.")
            
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            mean_absolute_error, mean_squared_error, r2_score, explained_variance_score,
            confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
        )
        
        y_pred = self.modelo.predict(X_test)
        y_pred_proba = self.modelo.predict_proba(X_test) if hasattr(self.modelo, 'predict_proba') else None
        
        resultados = {
            'tipo_problema': self.tipo_problema,
            'nombre_modelo': self.modelo.__class__.__name__,
            'hiperparametros': self.modelo.get_params()
        }
        
        if self.tipo_problema == 'clasificacion':
            # Métricas básicas
            resultados.update({
                'exactitud': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'sensibilidad': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'reporte_clasificacion': classification_report(y_test, y_pred, output_dict=True),
                'matriz_confusion': confusion_matrix(y_test, y_pred).tolist()
            })
            
            # Curva ROC y AUC si es posible
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:  # Solo para clasificación binaria
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                resultados['roc_auc'] = auc(fpr, tpr)
                resultados['curva_roc'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
                
                # Curva de precisión-exactitud
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
                resultados['curva_precision_recall'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist()
                }
            
            # Importancia de características si está disponible
            if hasattr(self.modelo, 'feature_importances_'):
                resultados['importancia_caracteristicas'] = self.modelo.feature_importances_.tolist()
            
        else:  # regresión
            resultados.update({
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'varianza_explicada': explained_variance_score(y_test, y_pred),
                'error_porcentual_absoluto_medio': np.mean(np.abs((y_test - y_pred) / np.maximum(1e-10, y_test))) * 100
            })
            
            # Gráficos de residuos
            residuos = y_test - y_pred
            resultados['residuos'] = {
                'media': float(np.mean(residuos)),
                'desviacion_estandar': float(np.std(residuos)),
                'min': float(np.min(residuos)),
                'max': float(np.max(residuos))
            }
            
            # Importancia de características si está disponible
            if hasattr(self.modelo, 'feature_importances_'):
                resultados['importancia_caracteristicas'] = self.modelo.feature_importances_.tolist()
            elif hasattr(self.modelo, 'coef_'):
                resultados['coeficientes'] = self.modelo.coef_.tolist()
        
        # Validación cruzada si se proporcionan datos de entrenamiento
        if X_train is not None and y_train is not None:
            from sklearn.model_selection import cross_val_score
            
            cv = min(5, len(np.unique(y_train)) if self.tipo_problema == 'clasificacion' else 5)
            scoring = 'accuracy' if self.tipo_problema == 'clasificacion' else 'r2'
            
            try:
                cv_scores = cross_val_score(
                    self.modelo, X_train, y_train,
                    cv=cv, scoring=scoring, n_jobs=-1
                )
                resultados['validacion_cruzada'] = {
                    'puntuaciones': cv_scores.tolist(),
                    'media': float(np.mean(cv_scores)),
                    'desviacion_estandar': float(np.std(cv_scores))
                }
            except Exception as e:
                logger.warning(f"No se pudo realizar validación cruzada: {str(e)}")
        
        return resultados
    
    def guardar_modelo(self, ruta: str):
        """
        Guarda el modelo y el preprocesador en disco
        
        Args:
            ruta: Ruta donde guardar el modelo
        """
        if self.modelo is None or self.preprocesador is None:
            raise ValueError("El modelo o el preprocesador no han sido inicializados")
            
        # Crear diccionario con el modelo y el preprocesador
        modelo_guardar = {
            'modelo': self.modelo,
            'preprocesador': self.preprocesador,
            'tipo_problema': self.tipo_problema,
            'columna_objetivo': self.columna_objetivo
        }
        
        # Guardar el modelo
        joblib.dump(modelo_guardar, ruta)
        logger.info(f"Modelo guardado en {ruta}")
    
    @staticmethod
    def cargar_modelo(ruta: str) -> Dict:
        """
        Carga un modelo guardado desde disco
        
        Args:
            ruta: Ruta al archivo del modelo
            
        Returns:
            Dict con el modelo y el preprocesador
        """
        return joblib.load(ruta)
