"""
Módulo de Modelado para la Clasificación de Estados Industriales.
Proporciona una interfaz unificada para modelos de Machine Learning.
"""

import joblib
import pandas as pd
from typing import Dict, Any, Optional, Union, Callable
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class StateClassifier:
    """Clasificador de estados operativos basado en firmas eléctricas."""

    def __init__(
        self,
        model_type: str = 'rf',
        params: Optional[Dict[str, Any]] = None,
        translator: Optional[Callable[[str], str]] = None
    ):
        """Inicializa el modelo con soporte para traducciones en errores."""
        self._ = translator or (lambda s: s)
        self.model_type = model_type
        self.params = params or self._get_default_params()
        self.scaler = StandardScaler()
        self.clf = self._initialize_model()

    def _get_default_params(self) -> Dict[str, Any]:
        """Parámetros optimizados para el entorno industrial."""
        if self.model_type == 'rf':
            return {"n_estimators": 100, "criterion": "entropy", "n_jobs": 8}
        return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}

    def _initialize_model(self) -> Union[RandomForestClassifier, XGBClassifier]:
        """Instancia el algoritmo seleccionado."""
        if self.model_type == 'rf':
            return RandomForestClassifier(**self.params)
        elif self.model_type == 'xgb':
            return XGBClassifier(**self.params, objective='multi:softmax')
        else:
            raise ValueError(self._("Algoritmo {} no soportado.").format(self.model_type))

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Entrena el modelo y devuelve la precisión obtenida."""
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        return float(accuracy_score(y, self.clf.predict(X_scaled)))

    def predict(self, X: pd.DataFrame):
        """Realiza predicciones sobre nuevos datos analógicos."""
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)

    def save(self, file_path: str) -> None:
        """Serializa el modelo y el escalador."""
        data = {
            "model": self.clf,
            "scaler": self.scaler,
            "model_type": self.model_type
        }
        joblib.dump(data, file_path)

