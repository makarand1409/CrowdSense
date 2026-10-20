"""Generate a six-month synthetic mall footfall dataset."""

from pathlib import Path
import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "mall_footfall.csv"


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end, periods=183, freq="D")
    rows = []
    weather_options = ["Sunny", "Cloudy", "Rainy"]
    festival_days = {15, 45, 76, 108, 139, 170}

    for date in dates:
        dow = date.day_name()
        weekend = dow in {"Saturday", "Sunday"}
        holiday = date.dayofyear % 30 == 0 or date.dayofyear in festival_days
        weather = rng.choice(weather_options, p=[0.55, 0.30, 0.15])
        for slot in pd.date_range("09:00", "21:30", freq="30min"):
            hour = slot.hour + slot.minute / 60
            if hour < 11:
                shape = 0.45
            elif 12 <= hour < 14:
                shape = 1.15
            elif 17 <= hour < 19:
                shape = 1.45 if weekend else 1.25
            elif 19 <= hour < 21:
                shape = 1.10 if weekend else 0.90
            elif hour >= 21:
                shape = 0.60
            else:
                shape = 0.80
            base = 105 if weekend else 72
            if holiday:
                base *= rng.uniform(2.0, 3.0)
            weather_factor = {"Sunny": 1.0, "Cloudy": 0.93, "Rainy": 0.70}[weather]
            count = max(4, int(rng.normal(base * shape * weather_factor, base * 0.10)))
            rows.append({
                "date": date.date().isoformat(),
                "day_of_week": dow,
                "time_slot": slot.strftime("%I:%M %p").lstrip("0"),
                "weather": weather,
                "is_holiday": bool(holiday),
                "is_weekend": bool(weekend),
                "footfall_count": count,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    generate_dataset().to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {OUTPUT_PATH}")
