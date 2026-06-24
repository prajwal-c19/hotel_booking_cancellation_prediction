"""
Hotel Booking Cancellation — Training Script
Run this ONCE (or whenever HotelData.xlsx changes) to train all models
and save the best one + chart metrics to disk for the Flask app to use.

Usage:
    python train_model.py
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────

df = pd.read_excel("HotelData.xlsx")
print(f"Loaded dataset: {df.shape}")

df["target"] = (df["booking_status"] == "Canceled").astype(int)

le = LabelEncoder()
df["meal_plan_enc"] = le.fit_transform(df["type_of_meal_plan"])
meal_plan_map = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
print(f"Meal plan encoding: {meal_plan_map}")

FEATURES = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "meal_plan_enc",
    "required_car_parking_space",
    "lead_time",
    "arrival_year",
    "arrival_month",
    "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
]

X = df[FEATURES]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 2. TRAIN ALL MODELS
# ─────────────────────────────────────────────

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0
    ),
}

results = {}
print("\nTraining models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    results[name] = {
        "model": model,
        "accuracy": float(acc),
        "auc": float(auc),
        "cm": cm.tolist(),
        # Downsample ROC curve points so the JSON stays small
        "roc_fpr": fpr[::max(1, len(fpr) // 100)].tolist(),
        "roc_tpr": tpr[::max(1, len(tpr) // 100)].tolist(),
    }
    print(f"  {name:22s} acc={acc*100:5.2f}%  auc={auc:.4f}")

# ─────────────────────────────────────────────
# 3. PICK BEST MODEL (by accuracy)
# ─────────────────────────────────────────────

best_name = max(results, key=lambda n: results[n]["accuracy"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)")

fi = pd.Series(
    best_model.feature_importances_ if hasattr(best_model, "feature_importances_") else np.zeros(len(FEATURES)),
    index=FEATURES,
).sort_values(ascending=False)

readable = {
    "no_of_special_requests": "Special requests",
    "repeated_guest": "Repeated guest",
    "required_car_parking_space": "Car parking space",
    "arrival_year": "Arrival year",
    "lead_time": "Lead time",
    "meal_plan_enc": "Meal plan",
    "avg_price_per_room": "Avg price/room",
    "arrival_month": "Arrival month",
    "no_of_previous_bookings_not_canceled": "Prev bookings (ok)",
    "no_of_previous_cancellations": "Prev cancellations",
    "no_of_weekend_nights": "Weekend nights",
    "no_of_adults": "No. of adults",
    "no_of_week_nights": "Week nights",
    "no_of_children": "No. of children",
    "arrival_date": "Arrival date",
}

# ─────────────────────────────────────────────
# 4. SAVE MODEL (pickle) + METRICS (json)
# ─────────────────────────────────────────────

with open("model.pkl", "wb") as f:
    pickle.dump({"model": best_model, "features": FEATURES, "meal_plan_map": meal_plan_map}, f)

metrics = {
    "best_model_name": best_name,
    "features": FEATURES,
    "meal_plan_map": meal_plan_map,
    "models": {
        name: {
            "accuracy": r["accuracy"],
            "auc": r["auc"],
            "cm": r["cm"],
            "roc_fpr": r["roc_fpr"],
            "roc_tpr": r["roc_tpr"],
        }
        for name, r in results.items()
    },
    "feature_importance": [
        {"feature": readable.get(f, f), "importance": round(float(v), 4)}
        for f, v in fi.items()
    ],
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved model.pkl and metrics.json")
print("Now run:  python app.py")