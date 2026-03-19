"""
AI-Powered Accident Hotspot Prediction
Team SENTINELS - STATATHON 2025
Data Generator: Synthetic Indian Road Accident Dataset
"""

import os
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from logger_config import setup_logger

logger = setup_logger(__name__)

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Indian cities with approximate bounding boxes
CITIES = {
    "Delhi": {"lat": (28.4, 28.9), "lon": (76.9, 77.4), "population": 32000000},
    "Mumbai": {"lat": (18.8, 19.3), "lon": (72.7, 73.1), "population": 21000000},
    "Bengaluru": {"lat": (12.8, 13.2), "lon": (77.4, 77.8), "population": 13000000},
    "Chennai": {"lat": (12.9, 13.3), "lon": (80.1, 80.3), "population": 11000000},
    "Hyderabad": {"lat": (17.2, 17.6), "lon": (78.3, 78.7), "population": 10000000},
    "Kolkata": {"lat": (22.4, 22.7), "lon": (88.2, 88.5), "population": 15000000},
    "Pune": {"lat": (18.4, 18.7), "lon": (73.7, 74.1), "population": 7000000},
    "Ahmedabad": {"lat": (22.9, 23.2), "lon": (72.4, 72.8), "population": 8000000},
}

# Hotspot zones (simulate high-risk corridors)
HOTSPOT_ZONES = {
    "Delhi": [(28.63, 77.22), (28.55, 77.10), (28.72, 77.35), (28.48, 77.08)],
    "Mumbai": [(19.07, 72.87), (18.92, 72.82), (19.15, 72.95), (18.85, 73.02)],
    "Bengaluru": [(12.97, 77.59), (12.85, 77.48), (13.07, 77.65), (12.92, 77.52)],
    "Chennai": [(13.08, 80.27), (12.95, 80.15), (13.15, 80.22), (13.02, 80.19)],
    "Hyderabad": [(17.38, 78.47), (17.45, 78.55), (17.28, 78.38), (17.52, 78.62)],
    "Kolkata": [(22.57, 88.36), (22.48, 88.28), (22.65, 88.43), (22.53, 88.32)],
    "Pune": [(18.52, 73.85), (18.60, 73.92), (18.45, 73.78), (18.57, 73.88)],
    "Ahmedabad": [(23.02, 72.57), (23.08, 72.63), (22.95, 72.50), (23.15, 72.70)],
}

ROAD_TYPES = ["National Highway", "State Highway", "Urban Road", "Rural Road", "Ring Road", "Expressway"]
WEATHER_CONDITIONS = ["Clear", "Rain", "Fog", "Haze", "Overcast", "Heavy Rain"]
VEHICLE_TYPES = ["Two-Wheeler", "Car/Jeep", "Truck/Lorry", "Bus", "Auto-Rickshaw", "Pedestrian"]
ACCIDENT_CAUSES = [
    "Overspeeding", "Drunk Driving", "Distracted Driving", "Poor Road Condition",
    "Signal Violation", "Wrong Side Driving", "Pedestrian Error", "Vehicle Failure",
    "Poor Visibility", "Pothole/Uneven Road"
]
SEVERITY_LABELS = ["Fatal", "Grievous Injury", "Minor Injury", "Property Damage"]


def generate_accident_near_hotspot(city, hotspot_center, radius=0.05):
    lat = hotspot_center[0] + np.random.normal(0, radius / 2)
    lon = hotspot_center[1] + np.random.normal(0, radius / 2)
    return lat, lon


def generate_synthetic_dataset(n_records=5000):
    records = []
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    for _ in range(n_records):
        city = random.choice(list(CITIES.keys()))
        city_info = CITIES[city]

        # 60% near hotspot, 40% random
        near_hotspot = random.random() < 0.60
        if near_hotspot and HOTSPOT_ZONES[city]:
            hotspot = random.choice(HOTSPOT_ZONES[city])
            lat, lon = generate_accident_near_hotspot(city, hotspot)
            # Clamp to city bounds
            lat = max(city_info["lat"][0], min(city_info["lat"][1], lat))
            lon = max(city_info["lon"][0], min(city_info["lon"][1], lon))
        else:
            lat = random.uniform(*city_info["lat"])
            lon = random.uniform(*city_info["lon"])

        # Temporal features
        accident_date = start_date + timedelta(days=random.randint(0, date_range))
        probs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.07, 0.06, 0.04,
               0.04, 0.05, 0.05, 0.04, 0.04, 0.05, 0.06, 0.08, 0.08, 0.07,
               0.05, 0.04, 0.03, 0.02])
        probs = probs / probs.sum()
        hour = int(np.random.choice(range(24), p=probs))
        is_peak_hour = 1 if hour in [7, 8, 9, 17, 18, 19, 20] else 0
        is_weekend = 1 if accident_date.weekday() >= 5 else 0
        month = accident_date.month
        is_monsoon = 1 if month in [6, 7, 8, 9] else 0

        # Weather weighted by monsoon
        if is_monsoon:
            weather = random.choices(
                WEATHER_CONDITIONS,
                weights=[25, 40, 5, 10, 15, 5]
            )[0]
        else:
            weather = random.choices(
                WEATHER_CONDITIONS,
                weights=[55, 10, 15, 10, 8, 2]
            )[0]

        road_type = random.choice(ROAD_TYPES)
        vehicle = random.choice(VEHICLE_TYPES)
        cause = random.choice(ACCIDENT_CAUSES)

        # Risk score calculation (0-100)
        risk = 30
        if near_hotspot:
            risk += 30
        if is_peak_hour:
            risk += 15
        if weather in ["Fog", "Heavy Rain", "Rain"]:
            risk += 15
        if road_type in ["National Highway", "Expressway"]:
            risk += 10
        if vehicle == "Two-Wheeler":
            risk += 10
        if cause in ["Overspeeding", "Drunk Driving"]:
            risk += 10
        risk = min(100, risk + random.randint(-10, 10))

        # Severity weighted by risk
        if risk > 70:
            severity = random.choices(SEVERITY_LABELS, weights=[40, 35, 15, 10])[0]
        elif risk > 50:
            severity = random.choices(SEVERITY_LABELS, weights=[20, 35, 30, 15])[0]
        else:
            severity = random.choices(SEVERITY_LABELS, weights=[10, 20, 40, 30])[0]

        num_killed = 0
        num_injured = 0
        if severity == "Fatal":
            num_killed = random.randint(1, 4)
            num_injured = random.randint(0, 5)
        elif severity == "Grievous Injury":
            num_injured = random.randint(1, 8)
        elif severity == "Minor Injury":
            num_injured = random.randint(1, 3)

        road_condition = random.choices(
            ["Good", "Average", "Poor", "Very Poor"],
            weights=[40, 30, 20, 10]
        )[0]
        lighting = random.choices(
            ["Daylight", "Artificial Light", "No Light"],
            weights=[60, 25, 15]
        )[0]
        speed_limit = random.choice([40, 50, 60, 80, 100, 120])
        estimated_speed = speed_limit + random.randint(-10, 40)

        records.append({
            "accident_id": f"ACC{_+1:05d}",
            "city": city,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "date": accident_date.strftime("%Y-%m-%d"),
            "year": accident_date.year,
            "month": month,
            "day_of_week": accident_date.strftime("%A"),
            "hour": hour,
            "is_peak_hour": is_peak_hour,
            "is_weekend": is_weekend,
            "is_monsoon": is_monsoon,
            "weather": weather,
            "road_type": road_type,
            "road_condition": road_condition,
            "lighting_condition": lighting,
            "vehicle_type": vehicle,
            "accident_cause": cause,
            "severity": severity,
            "num_killed": num_killed,
            "num_injured": num_injured,
            "speed_limit_kmph": speed_limit,
            "estimated_speed_kmph": estimated_speed,
            "risk_score": risk,
            "near_hotspot": int(near_hotspot),
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    try:
        logger.info("Starting synthetic accident dataset generation...")
        df = generate_synthetic_dataset(5000)
        
        if df is None or len(df) == 0:
            raise ValueError("Dataset generation produced empty or None result")
        
        output_path = os.path.join(DATA_DIR, "accidents.csv")
        df.to_csv(output_path, index=False)
        
        logger.info(f"[OK] Dataset generation complete: {len(df)} records")
        logger.info(f"[OK] Output saved to: {output_path}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.debug(f"Columns: {list(df.columns)}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        exit(1)
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during dataset generation: {e}", exc_info=True)
        exit(1)
