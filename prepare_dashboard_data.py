"""
Dashboard Data Preparation
Generates all JSON data needed by the HTML dashboard
"""
import json
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
STATIC_DIR = os.path.join(BASE, "static")
os.makedirs(STATIC_DIR, exist_ok=True)


def prepare_dashboard_data():
    # Load data
    df = pd.read_csv(f"{MODELS_DIR}/accidents_enriched.csv")
    clusters = pd.read_csv(f"{MODELS_DIR}/hotspot_clusters.csv")
    city_stats = pd.read_csv(f"{MODELS_DIR}/city_stats.csv")
    feat_imp = pd.read_csv(f"{MODELS_DIR}/feature_importance.csv")
    recs = pd.read_csv(f"{MODELS_DIR}/recommendations.csv")
    with open(f"{MODELS_DIR}/model_metrics.json") as f:
        metrics = json.load(f)

    # 1. Overview stats
    overview = {
        "total_accidents": int(len(df)),
        "total_fatalities": int(df["num_killed"].sum()),
        "total_injuries": int(df["num_injured"].sum()),
        "total_hotspots": int(len(clusters)),
        "cities_covered": int(df["city"].nunique()),
        "avg_risk_score": round(float(df["risk_score"].mean()), 1),
        "high_risk_count": int((df["risk_score"] >= 60).sum()),
        "model_accuracy": metrics["gradient_boosting"]["accuracy"],
        "model_auc": metrics["gradient_boosting"]["auc_roc"],
    }

    # 2. Map data - accidents sample (max 1000 for performance)
    map_sample = df.sample(min(1500, len(df)), random_state=42)
    map_accidents = map_sample[
        ["accident_id", "city", "latitude", "longitude", "severity",
         "risk_score", "weather", "road_type", "accident_cause",
         "hour", "num_killed", "num_injured", "predicted_risk_prob"]
    ].to_dict(orient="records")

    # 3. Hotspot markers
    hotspots = []
    for _, row in clusters.iterrows():
        rec_row = recs[(recs["city"] == row["city"]) &
                       (recs["cluster_id"] == row["cluster_id"])]
        rec_text = rec_row["recommendations"].values[0] if len(rec_row) > 0 else "Increase surveillance"
        risk_level = rec_row["risk_level"].values[0] if len(rec_row) > 0 else "MEDIUM"
        hotspots.append({
            "cluster_id": int(row["cluster_id"]),
            "city": str(row["city"]),
            "lat": float(row["center_lat"]),
            "lon": float(row["center_lon"]),
            "num_accidents": int(row["num_accidents"]),
            "num_fatal": int(row["num_fatal"]),
            "avg_risk": round(float(row["avg_risk_score"]), 1),
            "top_cause": str(row["top_cause"]),
            "top_weather": str(row["top_weather"]),
            "radius_km": round(float(row["radius_km"]), 2),
            "recommendation": str(rec_text),
            "risk_level": str(risk_level),
        })

    # 4. City statistics chart data
    city_chart = []
    for _, row in city_stats.iterrows():
        city_chart.append({
            "city": row["city"],
            "total_accidents": int(row["total_accidents"]),
            "total_fatalities": int(row["total_fatalities"]),
            "total_injuries": int(row["total_injuries"]),
            "avg_risk": round(float(row["avg_risk_score"]), 1),
            "fatality_rate": float(row["fatality_rate"]),
            "peak_hour_pct": round(float(row["peak_hour_accidents"] / row["total_accidents"] * 100), 1),
            "monsoon_pct": round(float(row["monsoon_accidents"] / row["total_accidents"] * 100), 1),
        })

    # 5. Time series - accidents by month/year
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.groupby([df["date"].dt.to_period("M").astype(str)]).agg(
        count=("accident_id", "count"),
        avg_risk=("risk_score", "mean"),
        fatalities=("num_killed", "sum")
    ).reset_index()
    monthly.columns = ["period", "count", "avg_risk", "fatalities"]
    time_series = monthly.to_dict(orient="records")

    # 6. Feature importance
    feat_data = feat_imp.head(12).to_dict(orient="records")

    # 7. Hourly distribution
    hourly = df.groupby("hour").agg(
        count=("accident_id", "count"),
        avg_risk=("risk_score", "mean")
    ).reset_index().to_dict(orient="records")

    # 8. Severity distribution
    severity_dist = df["severity"].value_counts().to_dict()

    # 9. Weather distribution
    weather_dist = df["weather"].value_counts().to_dict()

    # 10. Cause distribution
    cause_dist = df["accident_cause"].value_counts().head(8).to_dict()

    # 11. Road type risk
    road_risk = df.groupby("road_type")["risk_score"].mean().round(1).to_dict()

    # 12. Year-over-year trend
    yoy = df.groupby("year").agg(
        count=("accident_id", "count"),
        fatalities=("num_killed", "sum"),
        avg_risk=("risk_score", "mean")
    ).reset_index().to_dict(orient="records")

    # 13. Prediction scenarios
    scenarios = [
        {"name": "Clear Day, Normal Traffic", "risk": 32, "factors": ["Good weather", "Off-peak hours", "Good road"]},
        {"name": "Peak Hour, Urban Road", "risk": 55, "factors": ["Rush hour", "High traffic density", "Urban junction"]},
        {"name": "Night + Fog + Highway", "risk": 78, "factors": ["Poor visibility", "High speed", "Fatigue risk"]},
        {"name": "Monsoon + Poor Road", "risk": 85, "factors": ["Heavy rain", "Pothole-ridden", "Slippery surface"]},
        {"name": "Black Spot Zone", "risk": 94, "factors": ["Historical hotspot", "Multiple causes", "No signage"]},
    ]

    # Bundle everything
    dashboard_data = {
        "overview": overview,
        "map_accidents": map_accidents,
        "hotspots": hotspots,
        "city_chart": city_chart,
        "time_series": time_series,
        "feature_importance": feat_data,
        "hourly_dist": hourly,
        "severity_dist": severity_dist,
        "weather_dist": weather_dist,
        "cause_dist": cause_dist,
        "road_risk": road_risk,
        "yoy_trend": yoy,
        "scenarios": scenarios,
        "model_metrics": metrics,
    }

    with open(f"{STATIC_DIR}/dashboard_data.json", "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"Dashboard data prepared: {len(hotspots)} hotspots, {len(map_accidents)} accident points")
    return dashboard_data


if __name__ == "__main__":
    prepare_dashboard_data()
