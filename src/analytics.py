"""Analítica y métricas para FraudGuard Analytics."""

import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    """Calcula los KPIs principales del análisis de anomalías.

    Args:
        df: DataFrame con columnas 'status', 'amount', y 'prediction'.

    Returns:
        Diccionario con las métricas calculadas.
    """
    total = len(df)
    normales = len(df[df["status"] == "Normal"])
    sospechosas = len(df[df["status"] == "Sospechosa"])
    pct_anomalias = (sospechosas / total * 100) if total > 0 else 0
    monto_total = df["amount"].sum()
    monto_sospechoso = df.loc[df["status"] == "Sospechosa", "amount"].sum()

    return {
        "total_transacciones": total,
        "normales": normales,
        "sospechosas": sospechosas,
        "pct_anomalias": round(pct_anomalias, 2),
        "monto_total": monto_total,
        "monto_sospechoso": monto_sospechoso,
        "promedio_monto": df["amount"].mean(),
    }


def get_top_suspicious(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Retorna las top N transacciones más sospechosas ordenadas por score.

    Args:
        df: DataFrame con resultados del modelo.
        n: Número de transacciones a retornar.

    Returns:
        DataFrame con las transacciones más sospechosas.
    """
    suspicious = df[df["prediction"] == -1].copy()
    return suspicious.sort_values(
        "anomaly_score_normalized", ascending=False
    ).head(n)


def get_risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna la distribución de transacciones por nivel de riesgo.

    Args:
        df: DataFrame con la columna 'risk_level'.

    Returns:
        DataFrame con columnas 'Nivel' y 'Cantidad'.
    """
    counts = (
        df["risk_level"]
        .value_counts()
        .reindex(["Bajo", "Medio", "Alto", "Crítico"], fill_value=0)
    )
    return pd.DataFrame({"Nivel": counts.index, "Cantidad": counts.values})
