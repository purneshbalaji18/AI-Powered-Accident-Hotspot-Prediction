"""
run_all.py — One-click pipeline runner
Team SENTINELS | STATATHON 2025
"""
import os
import sys
import time
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))

steps = [
    ("1/3 — Generating Synthetic Dataset",       "data_generator.py"),
    ("2/3 — Running ML Pipeline",                "ml_pipeline.py"),
    ("3/3 — Preparing Dashboard Data (JSON)",    "prepare_dashboard_data.py"),
]

print("\n" + "="*60)
print("  SENTINELS — AI Accident Hotspot Prediction")
print("  STATATHON 2025 | PS-4 | Team 3346")
print("="*60 + "\n")

for label, script in steps:
    print(f"\n>>> {label}")
    print("-"*50)
    t0 = time.time()
    # Wrap paths in quotes to handle spaces in folder names
    cmd = [sys.executable, os.path.join(BASE, script)]
    print(f"Running: {cmd}")
    proc = subprocess.run(cmd, cwd=BASE)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\n[ERROR] {script} failed. Check above for details.")
        sys.exit(1)
    print(f"  [OK] Completed in {elapsed:.1f}s")

print("\n" + "="*60)
print("  ✅ All steps complete!")
print("="*60)
print("""
  Next Steps:
  ───────────
  1. Open dashboard.html in your browser:
       → Double-click dashboard.html
       → OR: python -m http.server 8080
              Visit: http://localhost:8080/dashboard.html

  2. Output files saved in:
       models/  — trained models, CSVs, JSONs
       static/  — dashboard_data.json
       data/    — accidents.csv

  Dashboard Tabs:
    📊 Overview      → KPI cards & trend charts
    🗺️  Hotspot Map   → Interactive map (zoom to hotspots!)
    📈 Analytics     → Deep dive charts
    🤖 Predictor     → Enter road conditions → get risk score
    🔬 Explainability → SHAP feature importance
    💡 Recommendations → AI safety interventions
    📋 Data Explorer  → Browse all 1500+ accident records
""")
