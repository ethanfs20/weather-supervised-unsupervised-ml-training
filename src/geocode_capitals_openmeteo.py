import os
import time
import requests
import pandas as pd

IN_CSV = "data/capitals.csv"
OUT_CSV = "data/capitals_geocoded.csv"

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


def geocode(name: str):
    params = {
        "name": name,
        "count": 5,
        "language": "en",
        "format": "json",
    }
    r = requests.get(GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("results")
    if not results:
        return None

    best = results[0]
    return (
        float(best["latitude"]),
        float(best["longitude"]),
        best.get("admin1", ""),
        best.get("country", ""),
    )


def main():
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv(IN_CSV)
    rows = []

    for _, r in df.iterrows():
        state = r["state"]
        capital = r["capital"]
        query = f"{capital}, {state}, USA"

        print("Geocoding:", query)
        g = geocode(query)
        if g is None:
            print("  FAILED")
            continue

        lat, lon, admin1, country = g
        rows.append(
            {
                "state": state,
                "capital": capital,
                "latitude": lat,
                "longitude": lon,
                "admin1": admin1,
                "country": country,
            }
        )
        print("  OK ->", lat, lon)
        time.sleep(0.2)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print("\nSaved:", OUT_CSV, "rows:", len(out))


if __name__ == "__main__":
    main()
