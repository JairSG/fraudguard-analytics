"""Funciones auxiliares para FraudGuard Analytics."""

import pandas as pd


# Columnas requeridas en el CSV de entrada
REQUIRED_COLUMNS = [
    "transaction_id",
    "date",
    "amount",
    "transaction_type",
    "merchant_category",
    "hour",
    "location_distance",
    "transactions_24h",
]

# Features numéricas que utiliza el modelo
FEATURE_COLUMNS = ["amount", "hour", "location_distance", "transactions_24h"]


def validate_csv(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Valida que el DataFrame tenga las columnas requeridas y tipos correctos.

    Args:
        df: DataFrame cargado del CSV.

    Returns:
        Tuple de (es_válido, lista_de_errores).
    """
    errors = []

    # Verificar columnas faltantes
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Columnas faltantes: {', '.join(missing)}")

    # Verificar que las features numéricas sean realmente numéricas
    for col in FEATURE_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"La columna '{col}' debe ser numérica")

    # Verificar que el dataset no esté vacío
    if len(df) == 0:
        errors.append("El dataset está vacío")

    return len(errors) == 0, errors


def format_currency(value: float) -> str:
    """Formatea un valor como moneda peruana.

    Args:
        value: Valor numérico.

    Returns:
        String formateado con prefijo S/.
    """
    return f"S/ {value:,.2f}"


def format_percentage(value: float) -> str:
    """Formatea un valor como porcentaje.

    Args:
        value: Valor numérico (ya en porcentaje).

    Returns:
        String formateado con símbolo %.
    """
    return f"{value:.2f}%"
