from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import gradio as gr

app = FastAPI()

# ======================
# LOAD MODEL 
# ======================
tfidf = joblib.load("model/tfidf.pkl")
similarity_df = joblib.load("model/similarity_df.pkl")

user_item = joblib.load("model/user_item.pkl")
user_item_centered = joblib.load("model/user_item_centered.pkl")
user_similarity_df = joblib.load("model/user_similarity.pkl")
user_mean = joblib.load("model/user_mean.pkl")

df_items = joblib.load("model/df_items.pkl")

# =========================
# UBCF
# =========================
def predict_single(user_id, item_id, k_neighbors=10):

    if user_id not in user_item_centered.index:
        return None

    if item_id not in user_item_centered.columns:
        return None

    sim_scores = user_similarity_df.loc[user_id].drop(user_id)
    sim_scores = sim_scores.sort_values(ascending=False).head(k_neighbors)

    numerator, denominator = 0, 0

    for other_user, sim in sim_scores.items():
        rating = user_item_centered.loc[other_user, item_id]

        if not np.isnan(rating):
            numerator += sim * rating
            denominator += sim

    if denominator == 0:
        return user_mean[user_id]

    return user_mean[user_id] + (numerator / denominator)

# =========================
# CBF
# =========================
def predict_cbf(user_id, item_id, train_df):

    user_data = train_df[train_df['users_id'] == user_id]

    if user_data.empty:
        return None

    liked_items = user_data.sort_values(
        'score', ascending=False
    )['culinary_places_id'].head(5)

    scores = []

    for liked_item in liked_items:
        if item_id in similarity_df.index and liked_item in similarity_df.columns:
            scores.append(similarity_df.loc[item_id, liked_item])

    return np.mean(scores) if scores else None

# =========================
# HYBRID
# =========================
def hybrid_recommend(user_id, top_n=5):

    predictions = {}

    for item in df_items['culinary_places_id']:

        ubcf = predict_single(user_id, item)
        cbf  = None  # bisa skip kalau ga ada train

        if ubcf is None:
            continue

        final_score = ubcf  # simple version dulu

        predictions[item] = final_score

    sorted_items = sorted(
        predictions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    ids = [x[0] for x in sorted_items]

    result = df_items[
        df_items['culinary_places_id'].isin(ids)
    ][['culinary_places_id', 'Title', 'Category', 'Rating']]

    return result.to_dict(orient="records")

# =========================
# API ENDPOINT
# =========================
@app.get("/")
def home():
    return {"message": "API Recommendation Running 🚀"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    return hybrid_recommend(user_id)

# =========================
# GRADIO WRAPPER
# =========================
def gradio_ui(user_id):
    return hybrid_recommend(int(user_id))

demo = gr.Interface(
    fn=gradio_ui,
    inputs="number",
    outputs="json",
    title="Hybrid Recommendation System"
)

# IMPORTANT
demo.launch()
