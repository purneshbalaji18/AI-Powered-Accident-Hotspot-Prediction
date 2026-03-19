# 🛡️ SENTINELS — AI-Powered Accident Hotspot Prediction
## STATATHON 2025 | Problem Statement PS-4 | Team 3346

---

## 📌 Overview

An end-to-end AI system that predicts accident-prone areas on Indian roads using:
- **Machine Learning** (Gradient Boosting + Random Forest)
- **DBSCAN Spatial Clustering** for hotspot detection
- **SHAP-based Explainability** for model transparency
- **Interactive Web Dashboard** with live map visualization
- **LLM-style Recommendations** for authorities

---

## 🗂️ Project Structure

```
accident_hotspot/
├── data_generator.py         # Synthetic dataset (5000 accidents, 8 cities)
├── ml_pipeline.py            # ML training pipeline
├── prepare_dashboard_data.py # Dashboard JSON export
├── dashboard.html            # 🌐 Full interactive web dashboard
├── run_all.py                # One-click run all steps
├── requirements.txt          # Python dependencies
│
├── data/
│   └── accidents.csv         # Generated accident dataset
│
└── models/
    ├── accidents_clustered.csv
    ├── accidents_enriched.csv
    ├── hotspot_clusters.csv
    ├── city_stats.csv
    ├── feature_importance.csv
    ├── recommendations.csv
    ├── model_metrics.json
    ├── shap_values.csv
    ├── gb_model.pkl           # Gradient Boosting model
    ├── rf_model.pkl           # Random Forest model
    └── scaler.pkl             # Feature scaler
```

---

## ⚙️ Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Everything
```bash
python run_all.py
```

### 3. Open Dashboard
Open `dashboard.html` in your browser — OR serve it:
```bash
cd accident_hotspot
python -m http.server 8080
# Visit: http://localhost:8080/dashboard.html
```

---

## 🤖 ML Pipeline

### Data Sources (Simulated/Substitutable)
| Source | Type | Used For |
|--------|------|----------|
| NCRB / MoRTH | Historical | Accident records |
| OpenStreetMap | GIS | Road network |
| OpenWeatherMap | Real-time | Weather conditions |
| Twitter/X API | Social | Live incident reports |

### Features (18 total)
- **Spatial**: Latitude, Longitude, Road Type, Road Condition
- **Temporal**: Hour, Day of Week, Month, Year, Peak Hour flag, Monsoon flag
- **Environmental**: Weather condition, Lighting
- **Behavioral**: Vehicle type, Accident cause, Speed

### Models
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Gradient Boosting (GBM) | 74.2% | 0.8079 |
| Random Forest | 70.1% | 0.7108 |

### Clustering (DBSCAN)
- **43 hotspot clusters** detected across 8 cities
- Uses Haversine metric (~500m radius)
- Handles irregular urban density

---

## 📊 Dashboard Features

| Tab | Description |
|-----|-------------|
| 📊 Overview | KPI cards, trend charts, severity distribution |
| 🗺️ Map | Interactive hotspot map with Leaflet.js (CartoDB dark tiles) |
| 📈 Analytics | Hourly patterns, weather impact, cause analysis |
| 🤖 Predictor | Real-time risk score from input parameters |
| 🔬 Explainability | SHAP feature importance, model comparison |
| 💡 Recommendations | AI-generated safety interventions per hotspot |
| 📋 Data Explorer | Filterable accident records table |

---

## 🏙️ Cities Covered
Delhi · Mumbai · Bengaluru · Chennai · Hyderabad · Kolkata · Pune · Ahmedabad

---

## 🔮 Innovation Highlights
1. **HDBSCAN/DBSCAN** dynamic clustering (vs static grid methods)
2. **SHAP Explainability** — no black box predictions
3. **Real-time Twitter NLP** for live hazard detection (architecture ready)
4. **LLM-style Recommendations** for each hotspot
5. **Cloud-ready** architecture with Kafka streaming support

---

## 📚 References
- Accident Prediction Models: https://arxiv.org/abs/2406.13968
- HDBSCAN: https://hdbscan.readthedocs.io/
- SHAP (Lundberg & Lee): https://arxiv.org/abs/1705.07874
- GNN Road Safety: https://arxiv.org/abs/2311.00164
- Deep Learning Crash Prediction: https://link.springer.com/article/10.1007/s44290-025-00255-3

---

**Team SENTINELS** | STATATHON 2025 | PS-4 | Team ID: 3346

