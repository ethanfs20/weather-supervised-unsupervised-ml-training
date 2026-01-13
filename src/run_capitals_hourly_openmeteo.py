import os
import time
from datetime import datetime, timezone

import requests
import pandas as pd

# -----------------------
# FILE PATHS
# -----------------------
IN_CSV = "data/capitals_geocoded.csv"
OUT_LOG = "reports/capitals_hourly_log.csv"

# -----------------------
# OPEN-METEO CONFIG
# -----------------------
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Simple rule-based thresholds (replace with ML later)
PROB_THRESHOLD = 50.0      # %
SUM_MM_THRESHOLD = 1.0     # mm

SLEEP_BETWEEN_CALLS = 0.1  # polite delay


def fetch_weather(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "UTC",
        "current": (
            "temperature_2m,"
            "precipitation,"
            "cloud_cover,"
            "wind_speed_10m,"
            "surface_pressure"
        ),
        "daily": (
            "temperature_2m_min,"
            "temperature_2m_max,"
            "precipitation_sum,"
            "precipitation_probability_max"
        ),
        "forecast_days": 2,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_daily(daily: dict, key: str, idx: int):
    arr = daily.get(key)
    if not isinstance(arr, list) or len(arr) <= idx:
        return None
    return arr[idx]


def decide_rain(prob_max, precip_sum):
    if prob_max is not None and float(prob_max) >= PROB_THRESHOLD:
        return "YES"
    if precip_sum is not None and float(precip_sum) >= SUM_MM_THRESHOLD:
        return "YES"
    return "NO"


def main():
    os.makedirs("reports", exist_ok=True)

    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Missing input file: {IN_CSV}")

    cities = pd.read_csv(IN_CSV)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = []

    for _, c in cities.iterrows():
        state = c["state"]
        capital = c["capital"]
        lat = float(c["latitude"])
        lon = float(c["longitude"])

        try:
            data = fetch_weather(lat, lon)
            cur = data.get("current", {}) or {}
            daily = data.get("daily", {}) or {}

            # index 1 = tomorrow
            tmin = get_daily(daily, "temperature_2m_min", 1)
            tmax = get_daily(daily, "temperature_2m_max", 1)
            pr_sum = get_daily(daily, "precipitation_sum", 1)
            pr_prob = get_daily(daily, "precipitation_probability_max", 1)

            pred = decide_rain(pr_prob, pr_sum)

            rows.append({
                "timestamp_utc": ts,
                "state": state,
                "capital": capital,
                "latitude": lat,
                "longitude": lon,

                "temp_c_now": cur.get("temperature_2m"),
                "precip_mm_now": cur.get("precipitation"),
                "cloud_pct_now": cur.get("cloud_cover"),
                "wind_kmh_now": cur.get("wind_speed_10m"),
                "pressure_hpa_now": cur.get("surface_pressure"),

                "tomorrow_temp_min_c": tmin,
                "tomorrow_temp_max_c": tmax,
                "tomorrow_precip_sum_mm": pr_sum,
                "tomorrow_precip_prob_max": pr_prob,

                "rain_tomorrow_pred": pred,
            })

        except Exception as e:
            rows.append({
                "timestamp_utc": ts,
                "state": state,
                "capital": capital,
                "latitude": lat,
                "longitude": lon,
                "error": str(e),
            })

        time.sleep(SLEEP_BETWEEN_CALLS)

    out = pd.DataFrame(rows)

    if os.path.exists(OUT_LOG):
        prev = pd.read_csv(OUT_LOG)
        out = pd.concat([prev, out], ignore_index=True)

    out.to_csv(OUT_LOG, index=False)

    print(f"Logged {len(rows)} cities at {ts}")
    print("Output:", OUT_LOG)


if __name__ == "__main__":
    main()
