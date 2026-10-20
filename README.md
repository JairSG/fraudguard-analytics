# 🛡️ FraudGuard Analytics

Aplicación web de analítica avanzada para la detección de transacciones financieras sospechosas mediante Machine Learning no supervisado.

<!-- ## Demo
🔗 [fraudguard-analytics.streamlit.app](https://fraudguard-analytics.streamlit.app/) -->

## Descripción

**FraudGuard Analytics** utiliza el algoritmo **Isolation Forest** para identificar patrones atípicos dentro de un conjunto de transacciones financieras. La aplicación no requiere datos etiquetados de fraude; aprende automáticamente qué transacciones son estadísticamente habituales y cuáles están aisladas respecto al resto.

## Características

- ✅ Carga de datasets CSV con validación automática
- ✅ Exploración de datos (EDA) con estadísticas y gráficos
- ✅ KPIs financieros en tiempo real
- ✅ Detección de anomalías con **Isolation Forest**
- ✅ **Anomaly Score** normalizado (0-100)
- ✅ **Niveles de riesgo**: Bajo → Medio → Alto → Crítico
- ✅ Ranking de operaciones sospechosas
- ✅ Detalle individual de transacciones
- ✅ Visualizaciones interactivas con Plotly
- ✅ Exportación de resultados a CSV
- ✅ Configuración interactiva de parámetros del modelo

## Tecnologías

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.12 |
| Frontend | Streamlit |
| Datos | Pandas + NumPy |
| Machine Learning | Scikit-learn (Isolation Forest) |
| Escalamiento | StandardScaler |
| Visualización | Plotly |
| Despliegue | Streamlit Community Cloud |

## Arquitectura

```
CSV
 ↓
Validación y Limpieza (Pandas)
 ↓
Selección de Features
 ↓
StandardScaler
 ↓
Isolation Forest
 ↓
predict() → Normal / Sospechosa
decision_function() → Anomaly Score
 ↓
Dashboard Interactivo (Plotly + Streamlit)
 ↓
Exportación CSV
```

## Estructura del Proyecto

```
FraudGuard Analytics/
│
├── app.py                  # Punto de entrada Streamlit
├── requirements.txt        # Dependencias fijadas
├── README.md               # Documentación
├── LICENSE                 # MIT License
├── .gitignore              # Python + C++
│
├── data/
│   └── sample_transactions.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Limpieza y escalado
│   ├── anomaly_detector.py # Isolation Forest
│   ├── analytics.py        # KPIs y ranking
│   └── utils.py            # Funciones auxiliares
│
├── assets/
│   └── screenshots/
│
└── .streamlit/
    └── config.toml         # Tema visual
```

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tuusuario/fraudguard-analytics.git
cd fraudguard-analytics

# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows)
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

## Dataset

El dataset de ejemplo (`data/sample_transactions.csv`) contiene 1,000 transacciones con las siguientes variables:

| Variable | Tipo | Descripción |
|---|---|---|
| `transaction_id` | string | Identificador único de la transacción |
| `date` | date | Fecha de la transacción |
| `amount` | float | Monto en S/ |
| `transaction_type` | string | Tipo: Purchase, Transfer, Withdrawal, Payment |
| `merchant_category` | string | Categoría: Food, Retail, Services, Entertainment, Other |
| `hour` | int | Hora del día (0-23) |
| `location_distance` | float | Distancia desde ubicación habitual (km) |
| `transactions_24h` | int | Cantidad de transacciones en últimas 24 horas |

### Variables utilizadas por el modelo

El modelo utiliza únicamente las siguientes variables numéricas:

- `amount` — Monto de la transacción
- `hour` — Hora del día
- `location_distance` — Distancia geográfica
- `transactions_24h` — Frecuencia de operaciones

## Modelo

### Isolation Forest

Isolation Forest es un algoritmo de **aprendizaje no supervisado** especializado en detección de anomalías. Funciona aislando observaciones mediante particiones aleatorias: las anomalías, al ser puntos raros y diferentes, requieren menos particiones para ser aisladas.

**Parámetros configurables:**

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `contamination` | 0.05 (5%) | Proporción esperada de anomalías |
| `n_estimators` | 100 | Número de árboles en el ensemble |
| `random_state` | 42 | Semilla para reproducibilidad |

### Pipeline

```
Datos → StandardScaler → IsolationForest → predict() + decision_function()
                                              ↓              ↓
                                        Normal/-1    Anomaly Score
                                              ↓
                                     Normalización 0-100
                                              ↓
                                     Nivel de Riesgo
```

## Resultados

La aplicación genera las siguientes métricas para cada transacción:

- **prediction**: 1 (Normal) o -1 (Sospechosa)
- **anomaly_score**: Score crudo del modelo
- **anomaly_score_normalized**: Score normalizado (0-100)
- **risk_level**: Clasificación interpretativa
  - 0-25: Bajo
  - 26-50: Medio
  - 51-75: Alto
  - 76-100: Crítico
- **status**: "Normal" o "Sospechosa"

## Limitaciones

> ⚠️ **Importante:** Esta aplicación identifica **anomalías estadísticas**, no fraude confirmado. Una transacción anómala no necesariamente representa un fraude real. Las transacciones marcadas como sospechosas requieren revisión humana adicional para determinar su naturaleza.

- El modelo no recibe una columna `fraude = sí/no`; aprende patrones normales del dataset.
- La clasificación de riesgo (Bajo/Medio/Alto/Crítico) es una interpretación del anomaly score, no una probabilidad de fraude.
- El rendimiento depende de la calidad y representatividad del dataset de entrada.
- Con datasets muy pequeños, el modelo puede no generalizar adecuadamente.

## Autor

**Jair Saavedra**

## Licencia

Este proyecto está bajo la [MIT License](LICENSE).

<!-- ## Capturas de Pantalla

### Dashboard
![Dashboard](assets/screenshots/dashboard.png)

### Detección de Anomalías
![Anomalías](assets/screenshots/anomalies.png)

### Transacciones Sospechosas
![Transacciones](assets/screenshots/transactions.png) -->
