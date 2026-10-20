"""
FraudGuard Analytics
====================
Aplicación web de analítica avanzada para la detección de transacciones
financieras sospechosas mediante Machine Learning no supervisado
utilizando Isolation Forest.

Ejecutar:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils import validate_csv, format_currency, format_percentage, FEATURE_COLUMNS
from src.preprocessing import clean_data, get_data_summary, scale_features
from src.anomaly_detector import run_isolation_forest
from src.analytics import compute_kpis, get_top_suspicious, get_risk_distribution


# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS personalizado
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Tarjetas de KPIs */
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.08);
        border: 1px solid rgba(28, 131, 225, 0.15);
        border-radius: 10px;
        padding: 15px 20px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem;
    }

    /* Encabezado principal */
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }

    /* Badge de estado */
    .badge-normal {
        background-color: #00C853;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-sospechosa {
        background-color: #FF1744;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-risk-bajo {
        background-color: #4CAF50;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-risk-medio {
        background-color: #FF9800;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-risk-alto {
        background-color: #F44336;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-risk-critico {
        background-color: #B71C1C;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Sidebar — Navegación
# ──────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shield.png",
    width=64,
)
st.sidebar.title("🛡️ FraudGuard")
st.sidebar.caption("Detector de anomalías financieras")
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegación",
    [
        "🏠 Inicio",
        "📂 Cargar Datos",
        "🔎 Exploración",
        "🤖 Detección",
        "📊 Dashboard",
        "🚨 Transacciones Sospechosas",
        "📥 Exportar Resultados",
    ],
)


# ──────────────────────────────────────────────
# Funciones auxiliares de UI
# ──────────────────────────────────────────────
def get_risk_badge(risk_level: str) -> str:
    """Retorna HTML de un badge de riesgo coloreado."""
    css_class = f"badge-risk-{risk_level.lower()}"
    return f'<span class="{css_class}">{risk_level}</span>'


def get_status_badge(status: str) -> str:
    """Retorna HTML de un badge de estado coloreado."""
    css_class = "badge-normal" if status == "Normal" else "badge-sospechosa"
    icon = "✅" if status == "Normal" else "🚨"
    return f'<span class="{css_class}">{icon} {status}</span>'


# ══════════════════════════════════════════════
# PÁGINA: Inicio
# ══════════════════════════════════════════════
if pagina == "🏠 Inicio":
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🛡️ FraudGuard Analytics")
    st.subheader("Detector inteligente de transacciones financieras sospechosas")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 Machine Learning")
        st.markdown(
            """
            - **Algoritmo:** Isolation Forest
            - **Tipo:** Aprendizaje no supervisado
            - **Objetivo:** Detección de anomalías
            - **Sin etiquetas:** No requiere datos de fraude previo
            """
        )

    with col2:
        st.markdown("### 📊 Analítica Avanzada")
        st.markdown(
            """
            - **KPIs** financieros en tiempo real
            - **Anomaly Score** normalizado (0-100)
            - **Ranking** de operaciones sospechosas
            - **Niveles de riesgo:** Bajo → Crítico
            """
        )

    st.divider()

    st.markdown("### 🔄 ¿Cómo funciona?")
    cols = st.columns(5)
    steps = [
        ("📂", "Datos", "Cargar CSV"),
        ("🔧", "Procesamiento", "Limpieza y escalado"),
        ("🤖", "Isolation Forest", "Detección de anomalías"),
        ("📈", "Anomaly Score", "Clasificación de riesgo"),
        ("📊", "Dashboard", "Visualización interactiva"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"#### {icon} {title}")
            st.caption(desc)

    st.divider()
    st.info(
        "⚠️ **Nota importante:** Esta aplicación identifica anomalías estadísticas. "
        "Una transacción anómala **no necesariamente representa un fraude confirmado**. "
        "Las transacciones marcadas como sospechosas requieren revisión adicional.",
        icon="ℹ️",
    )


# ══════════════════════════════════════════════
# PÁGINA: Cargar Datos
# ══════════════════════════════════════════════
elif pagina == "📂 Cargar Datos":
    st.title("📂 Cargar Dataset")
    st.markdown("Sube un archivo CSV con las transacciones a analizar.")

    uploaded_file = st.file_uploader(
        "Seleccionar archivo CSV",
        type=["csv"],
        help="El archivo debe contener las columnas: transaction_id, date, amount, "
        "transaction_type, merchant_category, hour, location_distance, transactions_24h",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            is_valid, errors = validate_csv(df)

            if is_valid:
                # Limpiar datos
                df = clean_data(df)
                st.session_state["df"] = df
                st.session_state["analysis_done"] = False

                st.success(f"✅ Archivo cargado correctamente", icon="✅")

                # Métricas de resumen
                summary = get_data_summary(df)
                col1, col2, col3 = st.columns(3)
                col1.metric("📋 Transacciones", f"{summary['total_registros']:,}")
                col2.metric("📊 Variables", summary["total_columnas"])
                col3.metric("❌ Valores nulos", summary["valores_nulos"])

                st.divider()

                # Vista previa
                st.markdown("### 👀 Vista previa de datos")
                st.dataframe(df.head(20), use_container_width=True, height=400)

                # Tipos de datos
                with st.expander("📋 Tipos de datos por columna"):
                    type_df = pd.DataFrame(
                        {
                            "Columna": df.dtypes.index,
                            "Tipo": df.dtypes.values.astype(str),
                            "No nulos": df.notnull().sum().values,
                            "Nulos": df.isnull().sum().values,
                        }
                    )
                    st.dataframe(type_df, use_container_width=True)

            else:
                st.error("❌ El archivo no tiene el formato esperado:")
                for error in errors:
                    st.warning(f"⚠️ {error}")

        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {e}")

    else:
        st.info(
            "💡 Puedes usar el dataset de ejemplo incluido en `data/sample_transactions.csv`",
            icon="📁",
        )

        # Botón para cargar dataset de ejemplo
        if st.button("📂 Cargar dataset de ejemplo", type="primary"):
            try:
                df = pd.read_csv("data/sample_transactions.csv")
                df = clean_data(df)
                st.session_state["df"] = df
                st.session_state["analysis_done"] = False
                st.success("✅ Dataset de ejemplo cargado correctamente")
                st.rerun()
            except FileNotFoundError:
                st.error("❌ No se encontró el archivo data/sample_transactions.csv")


# ══════════════════════════════════════════════
# PÁGINA: Exploración (EDA)
# ══════════════════════════════════════════════
elif pagina == "🔎 Exploración":
    st.title("🔎 Exploración de Datos")

    if "df" not in st.session_state:
        st.warning("⚠️ Primero debes cargar un dataset en la sección **📂 Cargar Datos**.")
        st.stop()

    df = st.session_state["df"]

    # KPIs de exploración
    st.markdown("### 📊 Resumen General")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Transacciones", f"{len(df):,}")
    col2.metric("💰 Monto Total", format_currency(df["amount"].sum()))
    col3.metric("📈 Promedio", format_currency(df["amount"].mean()))
    col4.metric("📊 Mediana", format_currency(df["amount"].median()))

    st.divider()

    # Gráficos EDA
    st.markdown("### 📈 Distribuciones")

    col1, col2 = st.columns(2)

    with col1:
        # 1. Distribución de montos
        fig_amount = px.histogram(
            df,
            x="amount",
            nbins=50,
            title="Distribución de Montos",
            labels={"amount": "Monto (S/)", "count": "Frecuencia"},
            color_discrete_sequence=["#667eea"],
        )
        fig_amount.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_amount, use_container_width=True)

    with col2:
        # 2. Transacciones por hora
        fig_hour = px.histogram(
            df,
            x="hour",
            nbins=24,
            title="Transacciones por Hora del Día",
            labels={"hour": "Hora", "count": "Frecuencia"},
            color_discrete_sequence=["#764ba2"],
        )
        fig_hour.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # 3. Transacciones por categoría
        category_counts = df["merchant_category"].value_counts().reset_index()
        category_counts.columns = ["Categoría", "Cantidad"]
        fig_cat = px.bar(
            category_counts,
            x="Categoría",
            y="Cantidad",
            title="Transacciones por Categoría",
            color="Categoría",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_cat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col4:
        # 4. Monto vs Distancia
        fig_scatter = px.scatter(
            df,
            x="location_distance",
            y="amount",
            title="Monto vs Distancia",
            labels={
                "location_distance": "Distancia (km)",
                "amount": "Monto (S/)",
            },
            color_discrete_sequence=["#f093fb"],
            opacity=0.6,
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Estadísticas descriptivas
    st.divider()
    st.markdown("### 📋 Estadísticas Descriptivas")
    st.dataframe(
        df[FEATURE_COLUMNS].describe().round(2),
        use_container_width=True,
    )


# ══════════════════════════════════════════════
# PÁGINA: Detección
# ══════════════════════════════════════════════
elif pagina == "🤖 Detección":
    st.title("🤖 Detección de Anomalías")

    if "df" not in st.session_state:
        st.warning("⚠️ Primero debes cargar un dataset en la sección **📂 Cargar Datos**.")
        st.stop()

    st.markdown(
        """
        Configura los parámetros del modelo **Isolation Forest** y ejecuta el análisis.
        El modelo identificará transacciones estadísticamente atípicas.
        """
    )

    # Configuración en sidebar
    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Configuración del Modelo")

    contamination = st.sidebar.slider(
        "Porcentaje esperado de anomalías",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="Proporción de transacciones que se espera sean anómalas.",
    )

    n_estimators = st.sidebar.slider(
        "Número de estimadores",
        min_value=50,
        max_value=300,
        value=100,
        step=10,
        help="Cantidad de árboles en el ensemble. Más árboles = mayor precisión pero más lento.",
    )

    random_state = st.sidebar.number_input(
        "Random State",
        min_value=0,
        max_value=9999,
        value=42,
        help="Semilla para reproducibilidad de resultados.",
    )

    # Resumen de configuración
    st.markdown("### ⚙️ Configuración Actual")
    config_col1, config_col2, config_col3 = st.columns(3)
    config_col1.metric("🎯 Contamination", format_percentage(contamination * 100))
    config_col2.metric("🌳 Estimadores", n_estimators)
    config_col3.metric("🎲 Random State", random_state)

    st.divider()

    # Features a utilizar
    st.markdown("### 📋 Variables del Modelo")
    st.markdown(
        """
        | Variable | Descripción |
        |---|---|
        | `amount` | Monto de la transacción |
        | `hour` | Hora del día (0-23) |
        | `location_distance` | Distancia desde ubicación habitual (km) |
        | `transactions_24h` | Transacciones en las últimas 24 horas |
        """
    )

    st.divider()

    # Botón de ejecución
    if st.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True):
        with st.spinner("⏳ Ejecutando Isolation Forest..."):
            df = st.session_state["df"]

            result = run_isolation_forest(
                df,
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=random_state,
            )

            st.session_state["df_result"] = result
            st.session_state["analysis_done"] = True

        # Resultados
        kpis = compute_kpis(result)
        st.success("✅ Análisis completado exitosamente", icon="✅")

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Normales", f"{kpis['normales']:,}")
        col2.metric("🚨 Sospechosas", f"{kpis['sospechosas']:,}")
        col3.metric("📊 % Anomalías", format_percentage(kpis["pct_anomalias"]))

        st.info(
            "Navega a **📊 Dashboard** para ver los resultados completos o a "
            "**🚨 Transacciones Sospechosas** para ver el ranking.",
            icon="👉",
        )

    elif st.session_state.get("analysis_done"):
        st.success("✅ Análisis ya ejecutado. Navega al Dashboard para ver los resultados.")


# ══════════════════════════════════════════════
# PÁGINA: Dashboard
# ══════════════════════════════════════════════
elif pagina == "📊 Dashboard":
    st.title("📊 Dashboard de Resultados")

    if not st.session_state.get("analysis_done"):
        st.warning(
            "⚠️ Primero debes ejecutar el análisis en la sección **🤖 Detección**."
        )
        st.stop()

    df_result = st.session_state["df_result"]
    kpis = compute_kpis(df_result)

    # KPIs principales
    st.markdown("### 📈 Indicadores Clave")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total", f"{kpis['total_transacciones']:,}")
    col2.metric("✅ Normales", f"{kpis['normales']:,}")
    col3.metric("🚨 Sospechosas", f"{kpis['sospechosas']:,}")
    col4.metric("📊 % Anomalías", format_percentage(kpis["pct_anomalias"]))

    col5, col6 = st.columns(2)
    col5.metric("💰 Monto Total", format_currency(kpis["monto_total"]))
    col6.metric("⚠️ Monto Sospechoso", format_currency(kpis["monto_sospechoso"]))

    st.divider()

    # Gráfico principal: scatter de anomalías
    st.markdown("### 🔍 Detección de Transacciones Anómalas")

    color_map = {"Normal": "#00C853", "Sospechosa": "#FF1744"}

    fig_main = px.scatter(
        df_result,
        x="date",
        y="amount",
        color="status",
        color_discrete_map=color_map,
        hover_data=["transaction_id", "hour", "location_distance", "anomaly_score_normalized"],
        title="Transacciones: Normales vs Sospechosas",
        labels={
            "date": "Fecha",
            "amount": "Monto (S/)",
            "status": "Estado",
            "transaction_id": "ID",
            "hour": "Hora",
            "location_distance": "Distancia (km)",
            "anomaly_score_normalized": "Anomaly Score",
        },
        size="anomaly_score_normalized",
        size_max=15,
    )
    fig_main.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
    )
    st.plotly_chart(fig_main, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Distribución de riesgo
        risk_dist = get_risk_distribution(df_result)
        risk_colors = {
            "Bajo": "#4CAF50",
            "Medio": "#FF9800",
            "Alto": "#F44336",
            "Crítico": "#B71C1C",
        }
        fig_risk = px.bar(
            risk_dist,
            x="Nivel",
            y="Cantidad",
            title="Distribución por Nivel de Riesgo",
            color="Nivel",
            color_discrete_map=risk_colors,
        )
        fig_risk.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col2:
        # Anomalías por hora
        suspicious_df = df_result[df_result["status"] == "Sospechosa"]
        fig_hour = px.histogram(
            suspicious_df,
            x="hour",
            nbins=24,
            title="Anomalías Detectadas por Hora",
            labels={"hour": "Hora del Día", "count": "Cantidad"},
            color_discrete_sequence=["#FF1744"],
        )
        fig_hour.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    # Gráfico adicional: Monto vs Distancia coloreado por status
    st.divider()
    st.markdown("### 📐 Monto vs Distancia")
    fig_dist = px.scatter(
        df_result,
        x="location_distance",
        y="amount",
        color="status",
        color_discrete_map=color_map,
        hover_data=["transaction_id", "anomaly_score_normalized", "risk_level"],
        labels={
            "location_distance": "Distancia (km)",
            "amount": "Monto (S/)",
            "status": "Estado",
        },
        opacity=0.7,
    )
    fig_dist.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════
# PÁGINA: Transacciones Sospechosas
# ══════════════════════════════════════════════
elif pagina == "🚨 Transacciones Sospechosas":
    st.title("🚨 Transacciones Sospechosas")

    if not st.session_state.get("analysis_done"):
        st.warning(
            "⚠️ Primero debes ejecutar el análisis en la sección **🤖 Detección**."
        )
        st.stop()

    df_result = st.session_state["df_result"]

    # Top transacciones sospechosas
    st.markdown("### 🏆 Top Transacciones Más Sospechosas")

    n_top = st.slider("Cantidad a mostrar", 5, 50, 10)
    top_suspicious = get_top_suspicious(df_result, n=n_top)

    if len(top_suspicious) == 0:
        st.info("No se encontraron transacciones sospechosas con la configuración actual.")
        st.stop()

    # Tabla de ranking
    display_cols = [
        "transaction_id",
        "amount",
        "hour",
        "location_distance",
        "transactions_24h",
        "anomaly_score_normalized",
        "risk_level",
    ]
    display_df = top_suspicious[display_cols].copy()
    display_df.columns = [
        "ID",
        "Monto (S/)",
        "Hora",
        "Distancia (km)",
        "Ops. 24h",
        "Anomaly Score",
        "Nivel de Riesgo",
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Monto (S/)": st.column_config.NumberColumn(format="S/ %.2f"),
            "Anomaly Score": st.column_config.ProgressColumn(
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
        },
    )

    st.divider()

    # Detalle de una transacción
    st.markdown("### 🔍 Detalle de Transacción")

    suspicious_ids = top_suspicious["transaction_id"].tolist()
    selected_tx = st.selectbox(
        "Seleccionar transacción para inspeccionar:",
        suspicious_ids,
    )

    if selected_tx:
        tx = df_result[df_result["transaction_id"] == selected_tx].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**🆔 ID:** `{tx['transaction_id']}`")
            st.markdown(f"**📅 Fecha:** {tx['date']}")
            st.markdown(f"**💰 Monto:** {format_currency(tx['amount'])}")
            st.markdown(f"**🏷️ Tipo:** {tx['transaction_type']}")

        with col2:
            st.markdown(f"**🕐 Hora:** {int(tx['hour']):02d}:00")
            st.markdown(f"**📍 Distancia:** {tx['location_distance']:.1f} km")
            st.markdown(f"**🔄 Ops. 24h:** {int(tx['transactions_24h'])}")
            st.markdown(f"**🏪 Categoría:** {tx['merchant_category']}")

        with col3:
            score = tx["anomaly_score_normalized"]
            risk = tx["risk_level"]
            status = tx["status"]

            st.markdown(f"**📊 Anomaly Score:** `{score:.1f} / 100`")
            st.markdown(
                f"**⚠️ Nivel de Riesgo:** {get_risk_badge(risk)}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**🔖 Estado:** {get_status_badge(status)}",
                unsafe_allow_html=True,
            )

        st.divider()

        # Indicador visual del score
        st.markdown("#### 📊 Anomaly Score")
        st.progress(min(score / 100, 1.0))
        if score >= 75:
            st.error(
                f"⚠️ Score de anomalía **{score:.1f}/100** — Nivel **{risk}**. "
                "Esta transacción presenta patrones altamente atípicos y requiere revisión."
            )
        elif score >= 50:
            st.warning(
                f"Score de anomalía **{score:.1f}/100** — Nivel **{risk}**. "
                "Se recomienda revisión."
            )
        else:
            st.info(
                f"Score de anomalía **{score:.1f}/100** — Nivel **{risk}**."
            )


# ══════════════════════════════════════════════
# PÁGINA: Exportar Resultados
# ══════════════════════════════════════════════
elif pagina == "📥 Exportar Resultados":
    st.title("📥 Exportar Resultados")

    if not st.session_state.get("analysis_done"):
        st.warning(
            "⚠️ Primero debes ejecutar el análisis en la sección **🤖 Detección**."
        )
        st.stop()

    df_result = st.session_state["df_result"]
    kpis = compute_kpis(df_result)

    st.markdown("### 📋 Resumen del Análisis")
    col1, col2, col3 = st.columns(3)
    col1.metric("📋 Total", f"{kpis['total_transacciones']:,}")
    col2.metric("🚨 Sospechosas", f"{kpis['sospechosas']:,}")
    col3.metric("📊 % Anomalías", format_percentage(kpis["pct_anomalias"]))

    st.divider()

    st.markdown("### 📄 Columnas incluidas en la exportación")
    export_columns = [
        "transaction_id",
        "date",
        "amount",
        "transaction_type",
        "merchant_category",
        "hour",
        "location_distance",
        "transactions_24h",
        "prediction",
        "anomaly_score",
        "anomaly_score_normalized",
        "risk_level",
        "status",
    ]

    cols_df = pd.DataFrame(
        {
            "Columna": export_columns,
            "Descripción": [
                "Identificador único",
                "Fecha de la transacción",
                "Monto en S/",
                "Tipo de transacción",
                "Categoría del comercio",
                "Hora del día",
                "Distancia desde ubicación habitual (km)",
                "Transacciones en últimas 24h",
                "Predicción: 1 (Normal) o -1 (Sospechosa)",
                "Score crudo de anomalía",
                "Score normalizado (0-100)",
                "Nivel de riesgo interpretativo",
                "Estado: Normal o Sospechosa",
            ],
        }
    )
    st.dataframe(cols_df, use_container_width=True, hide_index=True)

    st.divider()

    # Opción de exportar solo sospechosas
    export_option = st.radio(
        "¿Qué deseas exportar?",
        ["📋 Todas las transacciones", "🚨 Solo transacciones sospechosas"],
    )

    if "Solo" in export_option:
        df_export = df_result[df_result["status"] == "Sospechosa"][export_columns]
        filename = "fraudguard_sospechosas.csv"
    else:
        df_export = df_result[export_columns]
        filename = "fraudguard_results.csv"

    csv_data = df_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"📥 Descargar {filename}",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    st.caption(f"Archivo: **{filename}** — {len(df_export):,} registros")


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("Desarrollado por **Jair Saavedra**")
st.sidebar.caption("🛡️ FraudGuard Analytics v1.0")
st.sidebar.caption("Isolation Forest · Scikit-learn")
