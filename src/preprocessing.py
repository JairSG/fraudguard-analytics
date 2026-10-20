"""Preprocesamiento de datos para FraudGuard Analytics."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import FEATURE_COLUMNS


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el DataFrame: elimina duplicados, maneja NaN y convierte tipos.

    Args:
        df: DataFrame crudo cargado del CSV.

    Returns:
        DataFrame limpio y listo para análisis.
    """
    df = df.copy()

    # Eliminar registros duplicados por transaction_id
    df = df.drop_duplicates(subset="transaction_id", keep="first")

    # Rellenar NaN numéricos con la mediana de cada columna
    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Convertir columna date a datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Asegurar tipos numéricos para features
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Retorna un resumen estadístico del DataFrame.

    Args:
        df: DataFrame a resumir.

    Returns:
        Diccionario con métricas de resumen.
    """
    return {
        "total_registros": len(df),
        "total_columnas": len(df.columns),
        "valores_nulos": int(df.isnull().sum().sum()),
        "duplicados": int(
            df.duplicated(subset="transaction_id").sum()
            if "transaction_id" in df.columns
            else 0
        ),
        "tipos_datos": df.dtypes.astype(str).to_dict(),
    }


def scale_features(
    df: pd.DataFrame, features: list[str] | None = None
) -> tuple[np.ndarray, StandardScaler]:
    """Escala las features numéricas con StandardScaler.

    Args:
        df: DataFrame con las features a escalar.
        features: Lista de columnas a escalar. Si es None, usa FEATURE_COLUMNS.

    Returns:
        Tuple de (datos_escalados ndarray, scaler_ajustado).
    """
    if features is None:
        features = FEATURE_COLUMNS

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler
