<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2193b0,100:6dd5ed&height=220&section=header&text=Rencangku%20Hybrid%20Recommendation%20API&fontSize=32&fontColor=ffffff&animation=fadeIn&fontAlignY=35"/>

<br>

<img src="https://readme-typing-svg.herokuapp.com?font=Inter&weight=600&size=22&pause=1000&color=2196F3&center=true&vCenter=true&width=700&lines=Hybrid+Recommendation+System;User-Based+%2B+Content-Based;Context-Aware+Weather+Integration;Flask+REST+API;Deployed+on+Vercel" />

<br><br>

<img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&style=for-the-badge">
<img src="https://img.shields.io/badge/Flask-API-black?logo=flask&style=for-the-badge">
<img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&style=for-the-badge">
<img src="https://img.shields.io/badge/Vercel-Deployment-black?logo=vercel&style=for-the-badge">
<img src="https://img.shields.io/badge/Status-Production-success?style=for-the-badge">

<br><br>

<b>Machine Learning API for Hybrid Culinary Recommendation System</b><br>
Built with Flask • Scikit-Learn • Context-Aware Weather

</div>

<hr>

<h2>🚀 Overview</h2>

<p>
<b>Rencangku Hybrid Recommendation API</b> adalah REST API berbasis Machine Learning
yang memberikan rekomendasi tempat kuliner menggunakan pendekatan:
</p>

<ul>
<li>🤝 User-Based Collaborative Filtering (UBCF)</li>
<li>🧠 Content-Based Filtering (CBF)</li>
<li>🌦️ Context-Aware (Weather Integration)</li>
<li>📍 Location-Based Personalization</li>
</ul>

<hr>

<h2>🧠 Recommendation Architecture</h2>

<div align="center">
<h3>Hybrid Score = α (UBCF) + (1 - α) (CBF)</h3>
<p>Default: <b>α = 0.6</b></p>
</div>

<h4>1️⃣ User-Based Collaborative Filtering</h4>
<p>Menghitung kemiripan antar user menggunakan cosine similarity.</p>

<h4>2️⃣ Content-Based Filtering</h4>
<p>Menggunakan TF-IDF vectorization pada kategori kuliner.</p>

<h4>3️⃣ Context-Aware Weather</h4>
<p>Terintegrasi dengan OpenWeather API untuk menyesuaikan rekomendasi berdasarkan kondisi:</p>

<ul>
<li>❄️ Dingin</li>
<li>☀️ Panas</li>
<li>🌤️ Semua</li>
</ul>

<hr>

<h2>🏗️ System Architecture</h2>

<pre>
Mobile App
     │
     ▼
Flask API (Vercel)
     │
     ├── Hybrid Recommender Model (.pkl)
     ├── Weather API (OpenWeather)
     └── Location Filtering
     │
     ▼
Personalized Recommendation List
</pre>

<hr>

<h2>📦 Tech Stack</h2>

<table>
<tr>
<td><b>Language</b></td>
<td>Python 3.10</td>
</tr>
<tr>
<td><b>Backend</b></td>
<td>Flask</td>
</tr>
<tr>
<td><b>Machine Learning</b></td>
<td>Scikit-Learn</td>
</tr>
<tr>
<td><b>Data Processing</b></td>
<td>Pandas, NumPy</td>
</tr>
<tr>
<td><b>Context Integration</b></td>
<td>OpenWeather API</td>
</tr>
<tr>
<td><b>Deployment</b></td>
<td>Vercel</td>
</tr>
</table>

<hr>

<h2>📁 Project Structure</h2>

<pre>
API_ML/
│
├── api/
│   └── index.py
│
├── hybrid_recommender_model.pkl
├── requirements.txt
├── vercel.json
└── README.md
</pre>

<hr>

<h2>📡 API Endpoint</h2>

<h3>POST /recommend</h3>

<h4>Request Body (JSON)</h4>

<pre>
{
  "user_id": 1,
  "user_lat": -6.9246,
  "user_lon": 106.9051,
  "alpha": 0.6,
  "top_n": 10
}
</pre>

<h4>Response</h4>

<pre>
{
    "recommendations": [
        {
            "Categorize_Weather": "semua",
            "Category": "indonesian",
            "Distance_Score": 0.0819728419520229,
            "Distance_km": 11.199162261389942,
            "Latitude": -6.8891401,
            "Longitude": 107.0001341,
            "Popularity_Score": 0.03207941483803553,
            "Price_range": "Rp 25.000 - 50.000",
            "Rating": 4.3,
            "Rating_Score": 0.86,
            "Rating_count": 307,
            "Title": "RM. Marem 1",
            "culinary_places_id": 16
        }
    ],
    "weather": "semua"
}
</pre>

<hr>

<p><b>Production-safe:<b></p>

<div align="center">

<h2>👩‍💻 Author</h2>

<b>ohyuna</b>
<br><br>

<a href="https://github.com/ohyuna56na">
<img src="https://img.shields.io/badge/GitHub-Profile-black?logo=github&style=for-the-badge">
</a>

<br><br>

⭐ If you find this project useful, please consider giving it a star!

<br><br>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2193b0,100:6dd5ed&height=120&section=footer"/>

</div>
