"""
AI-Powered Accident Hotspot Prediction
Team SENTINELS - STATATHON 2025
ML Pipeline: DBSCAN Clustering + Gradient Boosting Prediction + SHAP Explainability
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_auc_score
)
from logger_config import setup_logger

logger = setup_logger(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "models")
DATA_PATH = os.path.join(BASE, "data", "accidents.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. HOTSPOT CLUSTERING (DBSCAN)
# ─────────────────────────────────────────────
def run_clustering(df):
    print("\n=== DBSCAN Hotspot Clustering ===")
    results = {}

    all_clusters = []
    for city, group in df.groupby("city"):
        coords = group[["latitude", "longitude"]].values
        # eps in radians for haversine metric (~500m)
        eps_km = 0.5
        eps_rad = eps_km / 6371.0

        db = DBSCAN(eps=eps_rad, min_samples=5, algorithm='ball_tree', metric='haversine')
        labels = db.fit_predict(np.radians(coords))

        group = group.copy()
        group["cluster_id"] = labels
        group["city_cluster"] = group.apply(
            lambda r: f"{city}_C{r['cluster_id']}" if r['cluster_id'] != -1 else "noise", axis=1
        )
        all_clusters.append(group)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"  {city}: {n_clusters} hotspot clusters, {n_noise} noise points")

        # Summarize clusters
        city_clusters = []
        for cid in set(labels):
            if cid == -1:
                continue
            mask = labels == cid
            cluster_pts = group[group["cluster_id"] == cid]
            city_clusters.append({
                "cluster_id": int(cid),
                "city": city,
                "center_lat": float(cluster_pts["latitude"].mean()),
                "center_lon": float(cluster_pts["longitude"].mean()),
                "num_accidents": int(mask.sum()),
                "num_fatal": int((cluster_pts["severity"] == "Fatal").sum()),
                "avg_risk_score": float(cluster_pts["risk_score"].mean()),
                "top_cause": cluster_pts["accident_cause"].mode()[0],
                "top_weather": cluster_pts["weather"].mode()[0],
                "radius_km": float(
                    np.sqrt(
                        (cluster_pts["latitude"].std() ** 2 +
                         cluster_pts["longitude"].std() ** 2)
                    ) * 111
                ) if len(cluster_pts) > 1 else 0.1,
            })
        results[city] = city_clusters

    df_clustered = pd.concat(all_clusters, ignore_index=True)
    df_clustered.to_csv(f"{OUTPUT_DIR}/accidents_clustered.csv", index=False)

    # Save cluster summary
    all_cluster_list = [c for city_cs in results.values() for c in city_cs]
    pd.DataFrame(all_cluster_list).to_csv(f"{OUTPUT_DIR}/hotspot_clusters.csv", index=False)

    print(f"\nTotal hotspot clusters found: {len(all_cluster_list)}")
    return df_clustered, results


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    print("\n=== Feature Engineering ===")
    df = df.copy()

    cat_features = ["city", "weather", "road_type", "road_condition",
                    "lighting_condition", "vehicle_type", "accident_cause", "day_of_week"]
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [
        "city_enc", "weather_enc", "road_type_enc", "road_condition_enc",
        "lighting_condition_enc", "vehicle_type_enc", "accident_cause_enc",
        "day_of_week_enc", "hour", "is_peak_hour", "is_weekend", "is_monsoon",
        "month", "year", "speed_limit_kmph", "estimated_speed_kmph",
        "latitude", "longitude"
    ]

    # Target: high_risk (risk_score >= 60)
    df["high_risk"] = (df["risk_score"] >= 60).astype(int)
    # Severity numeric
    sev_map = {"Fatal": 3, "Grievous Injury": 2, "Minor Injury": 1, "Property Damage": 0}
    df["severity_num"] = df["severity"].map(sev_map)

    print(f"  Features: {len(feature_cols)}")
    print(f"  High-risk samples: {df['high_risk'].sum()} / {len(df)}")

    return df, feature_cols, encoders


# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
def train_models(df, feature_cols):
    print("\n=== Model Training ===")

    X = df[feature_cols].fillna(0)
    y = df["high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Gradient Boosting (LightGBM substitute)
    print("  Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5,
        min_samples_split=20, random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_preds = gb_model.predict(X_test)
    gb_proba = gb_model.predict_proba(X_test)[:, 1]

    print(f"\n  [Gradient Boosting] Accuracy: {accuracy_score(y_test, gb_preds):.4f}")
    print(f"  [Gradient Boosting] AUC-ROC:  {roc_auc_score(y_test, gb_proba):.4f}")
    print(classification_report(y_test, gb_preds))

    # Random Forest
    print("  Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    print(f"\n  [Random Forest] Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print(f"  [Random Forest] AUC-ROC:  {roc_auc_score(y_test, rf_proba):.4f}")

    # Feature importance (SHAP-like manual)
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gb": gb_model.feature_importances_,
        "importance_rf": rf_model.feature_importances_,
    }).sort_values("importance_gb", ascending=False)
    feature_importance.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
    print("\n  Top-10 Feature Importances:")
    print(feature_importance.head(10).to_string(index=False))

    # Save models
    with open(f"{OUTPUT_DIR}/gb_model.pkl", "wb") as f:
        pickle.dump(gb_model, f)
    with open(f"{OUTPUT_DIR}/rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save metrics
    metrics = {
        "gradient_boosting": {
            "accuracy": round(accuracy_score(y_test, gb_preds), 4),
            "auc_roc": round(roc_auc_score(y_test, gb_proba), 4),
            "n_estimators": 150,
            "learning_rate": 0.1,
        },
        "random_forest": {
            "accuracy": round(accuracy_score(y_test, rf_preds), 4),
            "auc_roc": round(roc_auc_score(y_test, rf_proba), 4),
            "n_estimators": 100,
        }
    }
    with open(f"{OUTPUT_DIR}/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predict risk scores for full dataset
    df["predicted_risk_prob"] = gb_model.predict_proba(df[feature_cols].fillna(0))[:, 1]
    df["predicted_high_risk"] = gb_model.predict(df[feature_cols].fillna(0))

    return gb_model, rf_model, scaler, feature_importance, metrics


# ─────────────────────────────────────────────
# 4. SHAP-STYLE ANALYSIS (manual approximation)
# ─────────────────────────────────────────────
def compute_shap_approximate(df, gb_model, feature_cols):
    print("\n=== SHAP-style Feature Analysis ===")

    # Compute permutation-based importance approximation
    X = df[feature_cols].fillna(0)
    baseline_preds = gb_model.predict_proba(X)[:, 1]

    shap_vals = {}
    for col in feature_cols[:10]:  # Top 10 for speed
        X_perm = X.copy()
        X_perm[col] = X_perm[col].sample(frac=1, random_state=42).values
        perm_preds = gb_model.predict_proba(X_perm)[:, 1]
        shap_vals[col] = float(np.mean(np.abs(baseline_preds - perm_preds)))

    shap_df = pd.DataFrame(
        list(shap_vals.items()), columns=["feature", "shap_importance"]
    ).sort_values("shap_importance", ascending=False)
    shap_df.to_csv(f"{OUTPUT_DIR}/shap_values.csv", index=False)
    print("  SHAP approximation saved.")
    return shap_df


# ─────────────────────────────────────────────
# 5. CITY STATISTICS
# ─────────────────────────────────────────────
def compute_city_stats(df):
    print("\n=== City-wise Statistics ===")
    stats = df.groupby("city").agg(
        total_accidents=("accident_id", "count"),
        total_fatalities=("num_killed", "sum"),
        total_injuries=("num_injured", "sum"),
        avg_risk_score=("risk_score", "mean"),
        fatal_accidents=("severity", lambda x: (x == "Fatal").sum()),
        monsoon_accidents=("is_monsoon", "sum"),
        peak_hour_accidents=("is_peak_hour", "sum"),
    ).reset_index()
    stats["fatality_rate"] = (stats["total_fatalities"] / stats["total_accidents"] * 100).round(2)
    stats.to_csv(f"{OUTPUT_DIR}/city_stats.csv", index=False)
    print(stats.to_string(index=False))
    return stats


# ─────────────────────────────────────────────
# 6. LLM-STYLE RECOMMENDATIONS (rule-based)
# ─────────────────────────────────────────────
def generate_recommendations(cluster_df):
    recommendations = []
    for _, row in cluster_df.iterrows():
        recs = []
        if row["avg_risk_score"] > 70:
            recs.append("Deploy permanent traffic police unit at this junction")
        if row["top_cause"] == "Overspeeding":
            recs.append("Install speed cameras and electronic speed limit boards")
        if row["top_cause"] == "Drunk Driving":
            recs.append("Set up regular sobriety checkpoints, especially on weekends")
        if row["top_cause"] == "Poor Road Condition":
            recs.append("Immediate road repair and resurfacing required")
        if row["top_cause"] == "Pedestrian Error":
            recs.append("Install pedestrian guardrails and overhead crossing bridges")
        if row["top_weather"] in ["Fog", "Haze"]:
            recs.append("Install fog lights, rumble strips, and visibility markers")
        if row["top_weather"] in ["Rain", "Heavy Rain"]:
            recs.append("Improve drainage, add anti-skid surface treatment")
        if row["num_fatal"] > 5:
            recs.append("Declare as Black Spot — require mandatory safety audit")
        if not recs:
            recs.append("Increase patrolling frequency and install CCTV surveillance")
        recommendations.append({
            "cluster_id": row["cluster_id"],
            "city": row["city"],
            "center_lat": row["center_lat"],
            "center_lon": row["center_lon"],
            "num_accidents": row["num_accidents"],
            "avg_risk_score": row["avg_risk_score"],
            "recommendations": " | ".join(recs[:3]),
            "risk_level": "CRITICAL" if row["avg_risk_score"] > 70
                         else "HIGH" if row["avg_risk_score"] > 55
                         else "MEDIUM"
        })
    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv(f"{OUTPUT_DIR}/recommendations.csv", index=False)
    print(f"\n  Generated {len(rec_df)} recommendations")
    return rec_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("  AI Accident Hotspot Prediction — ML Pipeline")
        logger.info("  Team SENTINELS | STATATHON 2025")
        logger.info("=" * 60)

        # Data loading with validation
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
        
        logger.info(f"Loading dataset from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        
        if df.empty:
            raise ValueError("Loaded dataset is empty")
        
        logger.info(f"[OK] Loaded {len(df)} accident records")

        # Pipeline execution
        logger.info("Starting clustering...")
        df_clustered, cluster_results = run_clustering(df)
        
        logger.info("Starting feature engineering...")
        df_feat, feature_cols, encoders = engineer_features(df_clustered)
        
        logger.info("Training models...")
        gb_model, rf_model, scaler, feat_imp, metrics = train_models(df_feat, feature_cols)
        
        logger.info("Computing SHAP values...")
        shap_df = compute_shap_approximate(df_feat, gb_model, feature_cols)
        
        logger.info("Computing city statistics...")
        city_stats = compute_city_stats(df_clustered)

        cluster_summary = pd.read_csv(f"{OUTPUT_DIR}/hotspot_clusters.csv")
        logger.info("Generating recommendations...")
        rec_df = generate_recommendations(cluster_summary)

        # Save enriched dataset
        enriched_path = f"{OUTPUT_DIR}/accidents_enriched.csv"
        df_feat.to_csv(enriched_path, index=False)
        logger.info(f"[OK] Enriched dataset saved to: {enriched_path}")

        logger.info("=" * 60)
        logger.info("  [SUCCESS] Pipeline Complete! All outputs saved to models/")
        logger.info("=" * 60)
        logger.info(f"Model Performance:")
        logger.info(f"  Gradient Boosting  Accuracy: {metrics['gradient_boosting']['accuracy']:.4f}  AUC: {metrics['gradient_boosting']['auc_roc']:.4f}")
        logger.info(f"  Random Forest      Accuracy: {metrics['random_forest']['accuracy']:.4f}  AUC: {metrics['random_forest']['auc_roc']:.4f}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in ML pipeline: {e}", exc_info=True)
        sys.exit(1)
