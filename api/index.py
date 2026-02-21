import pickle
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==============================
# LOAD MODEL HYBRID
# ==============================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "hybrid_recommender_model.pkl"
)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

df_items = model["df_items"]

# shortcut fungsi
hybrid_recommendation = model["hybrid_recommendation"]


# ==============================
# HELPER: GET WEATHER
# ==============================
import requests

def get_weather(lat, lon, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    res = requests.get(url, params=params)
    data = res.json()

    temp = data["main"]["temp"]

    if temp <= 22:
        return "dingin"
    elif temp >= 30:
        return "panas"
    return "semua"


# ==============================
# ENDPOINT REKOMENDASI
# ==============================
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json

    user_id = data.get("user_id")
    user_lat = data.get("user_lat")
    user_lon = data.get("user_lon")
    alpha = data.get("alpha", 0.6)
    top_n = data.get("top_n", 10)

    if user_id is None:
        return jsonify({"error": "user_id required"}), 400

    # weather
    api_key = "8c79ecf19f5084f74baa0c841a95214f"
    weather_condition = get_weather(user_lat, user_lon, api_key)

    # rekomendasi
    rec = hybrid_recommendation(
        user_id=user_id,
        user_lat=user_lat,
        user_lon=user_lon,
        weather_condition=weather_condition,
        alpha=alpha,
        n=top_n
    )

    # format JSON
    result = rec.to_dict(orient="records")

    return jsonify({
        "weather": weather_condition,
        "recommendations": result
    })


# ==============================
# HEALTH CHECK
# ==============================
@app.route("/")
def home():
    return "ML Backend Rencangku is running."


if __name__ == "__main__":
    app.run(debug=True)