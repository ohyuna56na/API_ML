from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# ==========================
# CONFIG
# ==========================

XANO_BASE_URL = "https://x8ki-letl-twmt.n7.xano.io/api:uLfma1G0"
WEATHER_API_KEY = "8c79ecf19f5084f74baa0c841a95214f"

# ==========================
# FETCH DATA FROM XANO
# ==========================

def fetch_xano(endpoint):
    res = requests.get(f"{XANO_BASE_URL}/{endpoint}")
    return res.json()

def get_all_data():
    culinary = pd.DataFrame(fetch_xano("culinary_places"))
    reviews = pd.DataFrame(fetch_xano("reviews"))
    interactions = pd.DataFrame(fetch_xano("user_interactions"))

    # Gunakan reviews sebagai rating utama
    if not reviews.empty:
        reviews = reviews.rename(columns={"rating": "score"})

    # Tambahkan implicit rating dari interaction (opsional)
    if not interactions.empty:
        interactions = interactions.rename(columns={
            "interaction_value": "score"
        })
        reviews = pd.concat([
            reviews[['users_id','culinary_places_id','score']],
            interactions[['users_id','culinary_places_id','score']]
        ])

    return culinary, reviews


# ==========================
# UTIL FUNCTIONS
# ==========================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
        np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val + 1e-8)


def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    res = requests.get(url).json()
    main = res['weather'][0]['main'].lower()

    if main in ['rain', 'thunderstorm', 'drizzle']:
        return "dingin"
    return "panas"


def weather_score(place_weather, user_weather):
    if place_weather == "semua":
        return 1
    if place_weather == user_weather:
        return 1
    return 0


# ==========================
# ROOT ENDPOINT (GET ONLY)
# ==========================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API ML Running",
        "endpoint": "/recommend (POST)"
    })

# ==========================
# RECOMMEND ENDPOINT (POST ONLY)
# ==========================

@app.route("/recommend", methods=["POST"])
def recommend():

    data = request.json
    user_id = data["user_id"]
    title = data["title"]
    user_lat = data["latitude"]
    user_lon = data["longitude"]
    top_n = data.get("top_n", 10)

    # Ambil data
    culinary, ratings = get_all_data()

    if culinary.empty:
        return jsonify({"error": "No culinary data"}), 400

    # =====================
    # CONTENT BASED
    # =====================

    culinary['cbf_text'] = (
        culinary['Category'].astype(str) + " " +
        culinary['Price_range'].astype(str) + " " +
        culinary['Categorize_Weather'].astype(str)
    )

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(culinary['cbf_text'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    if title not in culinary['Title'].values:
        return jsonify({"error": "Title not found"}), 404

    idx = culinary[culinary['Title'] == title].index[0]
    cbf_scores = cosine_sim[idx]
    cbf_dict = dict(zip(culinary['id'], cbf_scores))

    # =====================
    # UBCF
    # =====================

    if ratings.empty:
        return jsonify({"error": "No rating data"}), 400

    user_item_matrix = ratings.pivot_table(
        index='users_id',
        columns='culinary_places_id',
        values='score',
        fill_value=0
    )

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    user_weather = get_weather(user_lat, user_lon)

    results = []

    for item_id in user_item_matrix.columns:

        # ===== UBCF =====
        if user_id in user_item_matrix.index:
            sim_users = user_similarity_df.loc[user_id]
            ratings_col = user_item_matrix[item_id]

            mean_user = user_item_matrix.loc[user_id].mean()
            mean_all = user_item_matrix.mean(axis=1)

            num = np.sum(sim_users * (ratings_col - mean_all))
            den = np.sum(np.abs(sim_users)) + 1e-8

            ubcf_score = mean_user + num / den
        else:
            ubcf_score = user_item_matrix.values.mean()

        ubcf_norm = normalize(ubcf_score, 1, 5)

        # ===== CBF =====
        item_idx = culinary[culinary['id'] == item_id].index[0]
        cbf_norm = cbf_scores[item_idx]

        # ===== DISTANCE =====
        place = culinary[culinary['id'] == item_id].iloc[0]

        dist = haversine(
            user_lat,
            user_lon,
            place['Latitude'],
            place['Longitude']
        )

        dist_score = 1 / (dist + 1)
        dist_norm = normalize(dist_score, 0, 1)

        # ===== WEATHER =====
        w_score = weather_score(
            place['Categorize_Weather'],
            user_weather
        )

        # ===== FINAL SCORE =====
        final_score = (
            0.25 * ubcf_norm +
            0.25 * cbf_norm +
            0.25 * dist_norm +
            0.25 * w_score
        ) * 5

        results.append({
            "id": int(item_id),
            "title": place['Title'],
            "header_image": place['Header_image'],
            "rating": float(place['Rating']),
            "address": place['Address'],
            "score": round(float(final_score), 3)
        })

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    return jsonify({
        "user_weather": user_weather,
        "recommendations": results[:top_n]
    })


# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    app.run(debug=True)
