"""
Hotel Booking Cancellation — Flask App
Serves the trained model + metrics to the frontend.

Usage:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import json
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ─────────────────────────────────────────────
# LOAD MODEL + METRICS ONCE AT STARTUP
# ─────────────────────────────────────────────

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)
    MODEL = bundle["model"]
    FEATURES = bundle["features"]
    MEAL_PLAN_MAP = bundle["meal_plan_map"]

with open("metrics.json", "r") as f:
    METRICS = json.load(f)


@app.route("/")
def index():
    return render_template("index.html", meal_plan_map=MEAL_PLAN_MAP)


@app.route("/metrics")
def metrics():
    """Returns all chart data for the dashboard (model comparison, ROC, feature importance)."""
    return jsonify(METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    """Takes booking form data, returns cancellation probability."""
    data = request.get_json()

    try:
        booking = {
            "no_of_adults": int(data["no_of_adults"]),
            "no_of_children": int(data["no_of_children"]),
            "no_of_weekend_nights": int(data["no_of_weekend_nights"]),
            "no_of_week_nights": int(data["no_of_week_nights"]),
            "meal_plan_enc": int(data["meal_plan_enc"]),
            "required_car_parking_space": int(data["required_car_parking_space"]),
            "lead_time": int(data["lead_time"]),
            "arrival_year": int(data["arrival_year"]),
            "arrival_month": int(data["arrival_month"]),
            "arrival_date": int(data["arrival_date"]),
            "repeated_guest": int(data["repeated_guest"]),
            "no_of_previous_cancellations": int(data["no_of_previous_cancellations"]),
            "no_of_previous_bookings_not_canceled": int(data["no_of_previous_bookings_not_canceled"]),
            "avg_price_per_room": float(data["avg_price_per_room"]),
            "no_of_special_requests": int(data["no_of_special_requests"]),
        }
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    row = pd.DataFrame([booking])[FEATURES]
    prob = float(MODEL.predict_proba(row)[0, 1])
    label = "Canceled" if prob >= 0.5 else "Not Canceled"

    return jsonify({"prediction": label, "cancel_probability": round(prob, 4)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)