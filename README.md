<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
</head>
<body style="margin:0; font-family: Arial, Helvetica, sans-serif; background-color:#0f172a; color:#e2e8f0;">

<!-- Animated Banner -->
<div style="padding:60px 20px; text-align:center; background: linear-gradient(-45deg, #1e293b, #0f172a, #1e40af, #0f172a); background-size:400% 400%; animation: gradient 10s ease infinite;">
<h1 style="font-size:42px; margin-bottom:10px;">🍽️ Rencangku Hybrid Recommendation API</h1>
<p style="font-size:18px; opacity:0.9;">
Production-Ready Machine Learning API for Context-Aware Culinary Recommendation
</p>
</div>

<style>
@keyframes gradient {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}
.section {
  padding:40px 20px;
  max-width:1000px;
  margin:auto;
}
.card {
  background-color:#1e293b;
  padding:20px;
  border-radius:12px;
  margin-bottom:20px;
}
.badge img {
  margin:5px;
}
code {
  background:#0f172a;
  padding:5px 8px;
  border-radius:6px;
  color:#38bdf8;
}
pre {
  background:#0f172a;
  padding:15px;
  border-radius:10px;
  overflow-x:auto;
}
</style>

<!-- Badges -->
<div class="section" style="text-align:center;">
<div class="badge">
<img src="https://img.shields.io/badge/Python-3.10-blue?logo=python">
<img src="https://img.shields.io/badge/Flask-REST%20API-black?logo=flask">
<img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn">
<img src="https://img.shields.io/badge/Deployment-Vercel-black?logo=vercel">
<img src="https://img.shields.io/badge/Status-Production-success">
</div>
</div>

<!-- About -->
<div class="section">
<h2>🚀 Project Overview</h2>
<div class="card">
<p>
This project is a <b>Hybrid Recommendation System API</b> developed to deliver personalized culinary recommendations.
</p>
<p>
The system integrates:
</p>
<ul>
<li>🤝 User-Based Collaborative Filtering (UBCF)</li>
<li>🧠 Content-Based Filtering (TF-IDF)</li>
<li>🌦️ Real-Time Weather Context Integration</li>
<li>📍 Location-Aware Personalization</li>
</ul>
<p>
Designed for scalability and deployment-ready architecture using Flask and Vercel.
</p>
</div>
</div>

<!-- Architecture -->
<div class="section">
<h2>🧠 Recommendation Architecture</h2>
<div class="card">
<p><b>Hybrid Formula:</b></p>
<p style="font-size:18px; color:#38bdf8;">
Hybrid Score = α (UBCF) + (1 - α) (CBF)
</p>
<p>Default α = 0.6</p>
</div>

<div class="card">
<h3>1️⃣ User-Based Collaborative Filtering</h3>
<p>Computes similarity between users using cosine similarity on user-item interaction matrix.</p>

<h3>2️⃣ Content-Based Filtering</h3>
<p>Uses TF-IDF vectorization on culinary categories to compute item similarity.</p>

<h3>3️⃣ Context-Aware (Weather Integration)</h3>
<p>Integrates OpenWeather API to adjust recommendation relevance:</p>
<ul>
<li>❄️ Cold</li>
<li>☀️ Hot</li>
<li>🌤️ Neutral</li>
</ul>
</div>
</div>

<!-- Tech Stack -->
<div class="section">
<h2>🛠 Tech Stack</h2>
<div class="card">
<ul>
<li>🐍 Python</li>
<li>🌐 Flask</li>
<li>📊 Pandas & NumPy</li>
<li>🧮 Scikit-Learn</li>
<li>☁️ OpenWeather API</li>
<li>🚀 Vercel Deployment</li>
</ul>
</div>
</div>

<!-- API -->
<div class="section">
<h2>📡 API Endpoint</h2>
<div class="card">
<p><b>POST</b> <code>/recommend</code></p>

<p><b>Request Body:</b></p>
<pre>{
  "user_id": 1,
  "user_lat": -6.9246,
  "user_lon": 106.9051,
  "alpha": 0.6,
  "top_n": 10
}</pre>

<p><b>Response:</b></p>
<pre>[
  {
    "culinary_places_id": 12,
    "name": "Warung Makan A",
    "score": 0.87
  }
]</pre>
</div>
</div>

<!-- Installation -->
<div class="section">
<h2>⚙️ Local Setup</h2>
<div class="card">
<pre>git clone https://github.com/ohyuna56na/API_ML.git
cd API_ML
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python api/index.py</pre>

<p>API runs at:</p>
<code>http://127.0.0.1:5000</code>
</div>
</div>

<!-- Portfolio Highlight -->
<div class="section">
<h2>🏆 Portfolio Highlights</h2>
<div class="card">
<ul>
<li>✔️ End-to-End ML Pipeline</li>
<li>✔️ Hybrid Recommendation Modeling</li>
<li>✔️ Context-Aware Personalization</li>
<li>✔️ REST API Deployment</li>
<li>✔️ Production-Safe Model Serialization</li>
</ul>
</div>
</div>

<!-- Academic -->
<div class="section">
<h2>🎓 Academic Research</h2>
<div class="card">
<p>
Developed as part of a thesis project:
</p>
<p><b>
Hybrid Recommendation System for Culinary Places using Context-Aware Approach
</b></p>
</div>
</div>

<!-- Author -->
<div class="section" style="text-align:center;">
<h2>👩‍💻 Author</h2>
<p><b>ohyuna</b></p>
</div>

<!-- Footer -->
<div style="text-align:center; padding:20px; background:#0f172a;">
<p>⭐ If this project interests you, feel free to connect!</p>
</div>

</body>
</html>
