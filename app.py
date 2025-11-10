from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Feature columns (must match training)
feature_cols = [
    "Channel ID", "Subscribers", "Views", "Videos",
    "Avg Views per Video", "Subscribers per Video", "Views per Subscriber",
    "Popularity_Score", "Video_Saturation",
    "Log_Subscribers", "Log_Views", "Log_Videos", "Posting_Intensity"
]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_page")
def predict_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        def safe_float(value):
            return float(value) if value not in [None, ""] else 0.0

        # User input
        subscribers = safe_float(request.form.get("Subscribers"))
        views = safe_float(request.form.get("Views"))
        videos = safe_float(request.form.get("Videos"))
        avg_views = safe_float(request.form.get("Avg_Views_per_Video"))

        # Derived features
        subs_per_video = subscribers / videos if videos else 0
        views_per_sub = views / subscribers if subscribers else 0
        popularity = 0
        saturation = 0
        intensity = 0
        log_subs = np.log1p(subscribers)
        log_views = np.log1p(views)
        log_videos = np.log1p(videos)
        channel_id = 9999

        # Prepare DataFrame
        input_df = pd.DataFrame([[
            channel_id, subscribers, views, videos,
            avg_views, subs_per_video, views_per_sub,
            popularity, saturation,
            log_subs, log_views, log_videos, intensity
        ]], columns=feature_cols)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict probability
        proba = model.predict_proba(input_scaled)[0][1] * 100

        # Categorize result
        if proba >= 15:
            result = "💜 HIGH ENGAGEMENT! You’re going viral! 🌟🎤"
        elif 8 <= proba < 15:
            result = "💫 MEDIUM ENGAGEMENT — Keep shining! ✨"
        else:
            result = "😴 LOW ENGAGEMENT — Try posting more consistently 💖"

        # Comparison data (scaled for demo so small user values show)
        comparison_data = {
            "user_subscribers": subscribers,
            "user_views": views,
            "user_videos": videos,
            "user_avg_views": avg_views,
            "avg_subscribers": max(1, subscribers*5),  # scaled average
            "avg_views": max(1, views*5),
            "avg_videos": max(1, videos*5),
            "avg_avg_views": max(1, avg_views*5)
        }

        return render_template(
            "index.html",
            prediction_text=result,
            comparison_data=comparison_data
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
