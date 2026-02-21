import pickle
import os
import pandas as pd
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from math import radians, cos, sin, asin, sqrt

app = Flask(__name__)
CORS(app)

# ==============================
# LOAD MODEL DATA ONLY
# ==============================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "hybrid_recommender_model.pkl"
)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

df_items = model["df_items"]
user_item = model["user_item"]
user_mean = model["user_mean"]
user_similarity_df = model["user_similarity_df"]
item_similarity_df = model["item_similarity_df"]


# ==============================
# WEATHER HELPER (WAJIB)
# ==============================
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
# FUNCTION UBCF
# ==============================
def predict_single(user_id, item_id, k_neighbors=5):

    if user_id not in user_item.index or item_id not in user_item.columns:
        return None

    sim_scores = user_similarity_df.loc[user_id].drop(user_id)
    sim_scores = sim_scores[sim_scores > 0].sort_values(ascending=False).head(k_neighbors)

    numerator = 0
    denominator = 0

    for other_user, sim in sim_scores.items():
        rating = user_item.loc[other_user, item_id]
        if not np.isnan(rating):
            numerator += sim * (rating - user_mean[other_user])
            denominator += sim

    if denominator == 0:
        return None

    return user_mean[user_id] + (numerator / denominator)


# ==============================
# FUNCTION CBF
# ==============================
def predict_cbf(user_id, item_id):
    user_history = user_item.loc[user_id].dropna()
    if len(user_history) == 0:
        return 0

    numerator = 0
    denominator = 0

    for interacted_item, score in user_history.items():
        if item_id in item_similarity_df.index:
            sim = item_similarity_df.loc[item_id, interacted_item]
            numerator += sim * score
            denominator += sim

    return numerator / denominator if denominator != 0 else 0


# ==============================
# HYBRID PREDICT
# ==============================
def hybrid_predict(user_id, item_id, user_lat, user_lon, weather_condition, alpha=0.6):

    ubcf_score = predict_single(user_id, item_id)
    cbf_score = predict_cbf(user_id, item_id)

    item_data = df_items[df_items["culinary_places_id"] == item_id]
    if item_data.empty:
        return None

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return 2 * asin(sqrt(a)) * R

    distance = haversine(
        user_lat, user_lon,
        item_data["Latitude"].values[0],
        item_data["Longitude"].values[0]
    )
    distance_score = 1 / (1 + distance)

    # WEATHER SCORE (WAJIB)
    item_weather = item_data["Categorize_Weather"].values[0]
    weather_score = 1 if (item_weather == weather_condition or item_weather == "semua") else 0

    if ubcf_score is None and cbf_score is None:
        return None

    if ubcf_score is None:
        base = cbf_score
    elif cbf_score is None:
        base = ubcf_score
    else:
        base = alpha * ubcf_score + (1 - alpha) * cbf_score

    return (0.6 * base) + (0.25 * distance_score) + (0.15 * weather_score)


# ==============================
# RECOMMENDATION ENDPOINT
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

    # WEATHER WAJIB
    api_key = "8c79ecf19f5084f74baa0c841a95214f"
    weather_condition = get_weather(user_lat, user_lon, api_key)

    predictions = {}

    for item in df_items["culinary_places_id"]:
        score = hybrid_predict(
            user_id,
            item,
            user_lat,
            user_lon,
            weather_condition,
            alpha
        )
        if score is not None:
            predictions[item] = score

    sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    if top_n is None or top_n == -1:
        item_ids = [x[0] for x in sorted_items]
    else:
        item_ids = [x[0] for x in sorted_items[:top_n]]

    result = df_items[df_items["culinary_places_id"].isin(item_ids)]

    return jsonify({
        "weather": weather_condition,
        "recommendations": result.to_dict(orient="records")
    })


@app.route("/")
def home():
    return "ML Backend Rencangku is running."


if __name__ == "__main__":
    app.run()