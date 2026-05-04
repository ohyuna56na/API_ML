from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# ======================
# LOAD MODEL (sekali saja saat start)
# ======================
tfidf = joblib.load("model/tfidf.pkl")
similarity_df = joblib.load("model/similarity_df.pkl")

user_item = joblib.load("model/user_item.pkl")
user_item_centered = joblib.load("model/user_item_centered.pkl")
user_similarity_df = joblib.load("model/user_similarity.pkl")
user_mean = joblib.load("model/user_mean.pkl")

df_items = joblib.load("model/df_items.pkl")

# ======================
# HYBRID PREDICT
# ======================
def hybrid_predict(user_id, item_id, k_neighbors=10):

    if user_id not in user_item.index:
        return None

    if item_id not in user_item.columns:
        return None

    # ===== UBCF =====
    sim_users = user_similarity_df.loc[user_id].drop(user_id)
    sim_users = sim_users.sort_values(ascending=False).head(k_neighbors)

    num, den = 0, 0

    for other_user, sim in sim_users.items():
        rating = user_item_centered.loc[other_user, item_id]

        if not np.isnan(rating):
            num += sim * rating
            den += sim

    if den == 0:
        ubcf_score = user_mean[user_id]
    else:
        ubcf_score = user_mean[user_id] + (num / den)

    # ===== CBF =====
    if item_id not in similarity_df.index:
        return None

    cbf_score = similarity_df.loc[item_id].mean()

    # ===== FINAL =====
    final_score = 0.7 * ubcf_score + 0.3 * cbf_score

    return float(final_score)

# ======================
# ENDPOINT
# ======================
@app.get("/")
def home():
    return {"message": "API Hybrid Recommendation Running 🚀"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int):

    predictions = {}

    for item in df_items['culinary_places_id']:

        # skip item yg sudah pernah diinteraksi
        if user_id in user_item.index:
            if item in user_item.columns:
                if not np.isnan(user_item.loc[user_id, item]):
                    continue

        score = hybrid_predict(user_id, item)

        if score is not None:
            predictions[item] = score

    # SORT
    top_items = sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    results = []

    for item_id, score in top_items:
        item = df_items[df_items['culinary_places_id'] == item_id].iloc[0]

        results.append({
            "id": int(item_id),
            "title": item["Title"],
            "category": item["Category"],
            "rating": float(item["Rating"]),
            "score": float(score)
        })

    return {
        "user_id": user_id,
        "recommendations": results
    }
