"""CrowdSense mall footfall analytics service."""

from datetime import datetime, timedelta
from pathlib import Path
import threading

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO
from sklearn.ensemble import RandomForestRegressor

from generate_data import generate_dataset, OUTPUT_PATH


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "data" / "footfall_model.joblib"
app = Flask(__name__)
app.config["SECRET_KEY"] = "crowdsense-live-analytics"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
data_lock = threading.Lock()
real_taps = []
counter_today = 0
last_tap_at = None


def load_dataset() -> pd.DataFrame:
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    if not OUTPUT_PATH.exists():
        generate_dataset().to_csv(OUTPUT_PATH, index=False)
    return pd.read_csv(OUTPUT_PATH)


dataset = load_dataset()
weather_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2}
day_map = {day: index for index, day in enumerate(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)}


def features(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["hour"] = frame["hour"].astype(float)
    result["day_of_week"] = frame["day_of_week"].map(day_map).fillna(0)
    result["is_weekend"] = frame["is_weekend"].astype(int)
    result["is_holiday"] = frame["is_holiday"].astype(int)
    result["weather"] = frame["weather"].map(weather_map).fillna(0)
    return result


def train_model() -> RandomForestRegressor:
    parsed = pd.to_datetime(dataset["time_slot"], format="%I:%M %p")
    train = dataset.copy()
    train["hour"] = parsed.dt.hour + parsed.dt.minute / 60
    model = RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=2, n_jobs=-1)
    model.fit(features(train), train["footfall_count"])
    joblib.dump(model, MODEL_PATH)
    return model


model = train_model()


def synthetic_summary() -> dict:
    hourly = dataset.copy()
    preview_dates = pd.to_datetime(dataset["date"])
    preview = dataset.loc[
        preview_dates.between("2026-10-01", "2026-10-10")
    ].head(12)
    parsed = pd.to_datetime(hourly["time_slot"], format="%I:%M %p")
    hourly.loc[:, "hour"] = parsed.dt.hour
    hourly_avg = hourly.groupby("hour", as_index=False)["footfall_count"].mean()
    hourly_avg.loc[:, "hour_label"] = hourly_avg["hour"].map(lambda value: f"{value:02d}:00")
    weekly = hourly.groupby("day_of_week", as_index=False)["footfall_count"].mean()
    weather = hourly.groupby("weather", as_index=False)["footfall_count"].mean()
    return {
        "hourly": hourly_avg.round(0).to_dict("records"),
        "weekly": weekly.round(0).to_dict("records"),
        "weather": weather.round(0).to_dict("records"),
        "peak_hours": hourly_avg.nlargest(3, "footfall_count").sort_values("hour").round(0).to_dict("records"),
        "dataset_preview": preview.to_dict("records"),
        "dataset_rows": len(dataset),
    }


@app.get("/")
def dashboard():
    return render_template("index.html")


@app.post("/tap")
def tap():
    global counter_today, last_tap_at
    payload = request.get_json(silent=True) or {}
    now = datetime.now()
    value = max(1, int(payload.get("count", 1)))
    tap_record = {"timestamp": now.isoformat(timespec="seconds"), "count": value}
    with data_lock:
        counter_today += value
        last_tap_at = now
        real_taps.append(tap_record)
        del real_taps[:-100]
    socketio.emit("new_tap", {"total_today": counter_today, **tap_record})
    return jsonify({"ok": True, **tap_record, "total_today": counter_today})


@app.get("/get_data")
def get_data():
    with data_lock:
        taps = list(real_taps)
        total = counter_today
        last_seen = last_tap_at.isoformat(timespec="seconds") if last_tap_at else None
    return jsonify({"taps": taps, "total_today": total, "last_tap": last_seen, "summary": synthetic_summary()})


@app.get("/get_synthetic")
def get_synthetic():
    return jsonify(synthetic_summary())


@app.get("/predict")
def predict():
    now = datetime.now()
    next_hour = now + timedelta(hours=1)
    weather = request.args.get("weather", "Sunny")
    sample = pd.DataFrame([{
        "hour": next_hour.hour,
        "day_of_week": next_hour.strftime("%A"),
        "is_weekend": next_hour.weekday() >= 5,
        "is_holiday": False,
        "weather": weather if weather in weather_map else "Sunny",
    }])
    prediction = float(model.predict(features(sample))[0])
    tree_predictions = np.array([tree.predict(features(sample))[0] for tree in model.estimators_])
    spread = float(np.std(tree_predictions))
    confidence = max(61, min(97, round(100 - (spread / max(prediction, 1) * 100), 1)))
    return jsonify({
        "predicted_footfall": round(prediction),
        "confidence": confidence,
        "window": next_hour.strftime("%I:%M %p").lstrip("0"),
        "weather": sample.at[0, "weather"],
    })


@app.post("/retrain")
def retrain():
    global model, dataset
    dataset = load_dataset()
    model = train_model()
    return jsonify({"ok": True, "message": "Random Forest model retrained", "rows": len(dataset)})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
