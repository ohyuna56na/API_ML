from flask import Flask, request, jsonify
import pickle
import pandas as pd
import requests

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
with open("model_recommender.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# XANO CONFIG
# =========================
BASE_URL = "https://x8ki-letl-twmt.n7.xano.io/api:uLfma1G0"

def fetch_xano(endpoint):
    url = f"{BASE_URL}/{endpoint}"
    res = requests.get(url)
    res.raise_for_status()
    return pd.DataFrame(res.json())

# =========================
# WEATHER FUNCTION
# =========================
api_key = "8c79ecf19f5084f74baa0c841a95214f"

def get_weather(lat, lon, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    data = response.json()

    temp = data["main"]["temp"]

    if temp <= 22:
        return "dingin"
    elif temp >= 30:
        return "panas"
    else:
        return "semua"

# =========================
# RECOMMENDATION ENDPOINT
# =========================
@app.route("/recommend", methods=["GET"])
def recommend():

    try:
        user_id = int(request.args.get("user_id"))
        user_lat = float(request.args.get("lat"))
        user_lon = float(request.args.get("lon"))
        api_key = request.args.get("weather_key")

        # =========================
        # GET DATA FROM XANO
        # =========================
        df_places  = fetch_xano("culinary_places")
        df_fav     = fetch_xano("favorites")
        df_reviews = fetch_xano("reviews")
        df_views   = fetch_xano("user_interactions")

        # =========================
        # DATA PREPARATION
        # =========================

        # Gabungkan interaksi jadi satu score
        df_reviews["score"] = df_reviews["rating"]

        df_views["score"] = 1
        df_fav["score"] = 2

        interactions = pd.concat([
            df_reviews[["user_id", "culinary_places_id", "score"]],
            df_views[["user_id", "culinary_places_id", "score"]],
            df_fav[["user_id", "culinary_places_id", "score"]]
        ])

        interactions = interactions.rename(
            columns={"user_id": "users_id"}
        )

        # =========================
        # UPDATE MODEL TRAIN DATA
        # =========================
        model.train = interactions
        model.df_items = df_places

        # =========================
        # WEATHER
        # =========================
        weather_condition = get_weather(
            user_lat,
            user_lon,
            api_key
        )

        # =========================
        # HYBRID RECOMMENDATION
        # =========================
        result = model.hybrid_recommendation(
            user_id=user_id,
            user_lat=user_lat,
            user_lon=user_lon,
            weather_condition=weather_condition,
            alpha=0.6,
            n=10
        )

        return jsonify(result.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# HEALTH CHECK
# =========================
@app.route("/")
def home():
    return jsonify({"status": "ML API Running 🚀"})