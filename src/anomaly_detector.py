"""Detector de anomalías con Isolation Forest para FraudGuard Analytics."""

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.preprocessing import scale_features
from src.utils import FEATURE_COLUMNS


def run_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Ejecuta Isolation Forest y agrega columnas de resultado al DataFrame.

    El modelo aprende qué transacciones son estadísticamente habituales y cuáles
    están aisladas respecto al resto. No recibe etiquetas de fraude.

    Columnas agregadas:
        - prediction: 1 (Normal) o -1 (Sospechosa)
        - anomaly_score: score crudo de decision_function()
        - anomaly_score_normalized: score normalizado 0-100 (100 = más anómalo)
        - risk_level: Bajo, Medio, Alto, Crítico
        - status: "Normal" o "Sospechosa"

    Args:
        df: DataFrame limpio con las features requeridas.
        contamination: Proporción esperada de anomalías (0.01 a 0.20).
        n_estimators: Número de árboles en el ensemble.
        random_state: Semilla para reproducibilidad.
        features: Lista de columnas a usar. Si es None, usa FEATURE_COLUMNS.

    Returns:
        DataFrame con las columnas de resultado agregadas.
    """
    if features is None:
        features = FEATURE_COLUMNS

    df = df.copy()

    # Escalar features
    X_scaled, _ = scale_features(df, features)

    # Entrenar y predecir
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )

    df["prediction"] = model.fit_predict(X_scaled)
    df["anomaly_score"] = model.decision_function(X_scaled)

    # Normalizar score a 0-100 (0 = completamente normal, 100 = muy anómalo)
    # decision_function retorna valores más negativos para anomalías,
    # por lo que invertimos la escala.
    raw_scores = df["anomaly_score"]
    min_score = raw_scores.min()
    max_score = raw_scores.max()

    if max_score != min_score:
        df["anomaly_score_normalized"] = (
            (1 - (raw_scores - min_score) / (max_score - min_score)) * 100
        ).round(1)
    else:
        df["anomaly_score_normalized"] = 50.0

    # Clasificación interpretativa del anomaly score
    df["risk_level"] = pd.cut(
        df["anomaly_score_normalized"],
        bins=[-0.1, 25, 50, 75, 100.1],
        labels=["Bajo", "Medio", "Alto", "Crítico"],
    )

    # Etiqueta legible
    df["status"] = df["prediction"].map({1: "Normal", -1: "Sospechosa"})

    return df
